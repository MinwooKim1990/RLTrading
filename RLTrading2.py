# %%
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
import os

# -----------------------------------------------
# GPU Acceleration Setup
# -----------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Using device:", device)

########################################
# 1. 데이터 수집 및 전처리 (90% Train, 10% Validation)
########################################
tickers = [
    "AAPL", "DLB", "DIS", "MSFT", "META", "BRK-A", "AVGO", "AMZN", "IONQ", "NVDA",
    "OKLO", "LUNR", "JPM", "CAKE", "KO", "TSLA", "TTWO", "HON", "ARM", "GOOGL"
]

data = yf.download(tickers, period="6mo", interval="1d")["Adj Close"]
data = data.sort_index().dropna()
split_idx = int(len(data) * 0.9)
train_data = data.iloc[:split_idx].reset_index(drop=True)
val_data   = data.iloc[split_idx:].reset_index(drop=True)

########################################
# 2. 환경: 개선된 주식 거래 환경 (Idle 액션, 거래 페널티, 보상 정규화/상대 보상)
########################################
class ImprovedStockTradingEnv:
    def __init__(self, price_data, tickers, initial_capital=10000, transaction_cost=0.01, window=5, 
                 max_trades_per_day=10, cash_fraction=0.1, trade_penalty=0.02):
        """
        price_data: 주가 데이터 (pandas DataFrame)
        tickers: 주식 티커 리스트
        initial_capital: 초기 자본
        cash_fraction: 초기 현금 비율
        trade_penalty: 거래가 실행될 때마다 부과되는 추가 페널티 (예: 0.02)
        """
        self.price_data = price_data.copy().reset_index(drop=True)
        self.num_days = len(self.price_data)
        self.num_stocks = self.price_data.shape[1]
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.window = window
        self.max_trades_per_day = max_trades_per_day
        self.cash_fraction = cash_fraction
        self.trade_penalty = trade_penalty  # 거래 시 추가 패널티

        # 포트폴리오: index 0 = 현금, indices 1..num_stocks = 주식 보유 수량
        self.num_assets = self.num_stocks + 1
        self.initial_prices = self.price_data.iloc[0].values  

        # 액션 매핑: 0번은 Idle 액션, 나머지는 매수/매도 거래 액션
        self.actions = self._create_action_mapping()
        self.action_space = list(self.actions.keys())
        self.action_size = len(self.actions)

        # 기록용 변수들
        self.daily_history = []   # 하루마다 포트폴리오 가치
        self.error_bars = []      # 당일 포트폴리오 상태 분산 (예: error bar 용)
        self.transaction_log = [] # 전체 거래 로그
        self.trades_today = []    # 해당 일에 실행한 거래 리스트
        self.daily_values = []    # 당일 중간 포트폴리오 가치 기록

        # [추가] 새로운 보상 관련 하이퍼파라미터
        self.diversification_bonus_weight = 0.05  # 주식 간 상관관계 보너스 가중치
        self.sharpe_bonus_weight = 0.1            # Sharpe 비율 기반 보너스 가중치
        self.long_term_bonus_weight = 0.2         # 장기 성장 보너스 가중치
        self.holding_period = 3                   # 장기 성과 평가 기간 (일 단위)
        self.risk_reward_weight = 0.1             # 기존 위험 보상 가중치
        self.daily_returns = []                   # Sharpe 보상 계산용 일간 수익률 기록

        self.reset()

    def _create_action_mapping(self):
        """
        액션 매핑 사전 생성  
        0: "idle" 액션 (거래 없음)
        그 외: (src, dst, fraction) 튜플  
             - src == 0: 매수 (현금 -> 주식)
             - dst == 0: 매도 (주식 -> 현금)
        """
        actions = {}
        actions[0] = "idle"  # Idle 액션
        action_id = 1
        fractions = [0.25, 0.5, 0.75, 1.0]
        # 매수 액션: 현금에서 주식으로
        for stock in range(1, self.num_assets):
            for f in fractions:
                actions[action_id] = (0, stock, f)
                action_id += 1
        # 매도 액션: 주식에서 현금으로
        for stock in range(1, self.num_assets):
            for f in fractions:
                actions[action_id] = (stock, 0, f)
                action_id += 1
        return actions

    def reset(self):
        self.current_day = 0
        self.portfolio = np.zeros(self.num_assets)
        self.portfolio[0] = self.initial_capital  # 전액 현금으로 시작
        initial_value = self.get_total_value(self.price_data.iloc[self.current_day].values)
        self.day_start_value = initial_value  # 하루 시작 시 포트폴리오 가치 저장
        self.daily_history = [initial_value]
        self.error_bars = [0.0]
        self.transaction_log = []
        self.daily_values = [initial_value]
        self.trades_today = []
        self.day_start_prices = self.price_data.iloc[self.current_day].values.copy()
        self.cost_basis = np.zeros(self.num_assets)  # 각 주식의 평균 매입 단가 추적 (현금 제외)
        self.daily_returns = []  # 일일 수익률 기록 초기화
        return self._get_state()

    def _get_state(self):
        """
        상태 구성: 포트폴리오 비율 + 기술적 피처  
        기술적 피처: 가격비율, 이동평균비, 변동성, Black-Scholes 비율, 평균 수익률
        """
        if self.current_day < self.num_days:
            current_prices = self.price_data.iloc[self.current_day].values
        else:
            current_prices = self.price_data.iloc[-1].values
        stock_values = np.array([self.portfolio[i] * current_prices[i - 1] for i in range(1, self.num_assets)])
        total_value = self.portfolio[0] + np.sum(stock_values)
        portfolio_frac = np.concatenate(([self.portfolio[0] / total_value], stock_values / total_value))
        price_ratio = current_prices / self.initial_prices
        
        # 기술적 피처 계산
        ma_ratios = []
        volatilities = []
        bs_ratios = []
        avg_returns = []
        T_const = 30 / 365
        r = 0.01
        for i in range(self.num_stocks):
            start_idx = max(0, self.current_day - self.window + 1)
            window_prices = self.price_data.iloc[start_idx:self.current_day+1, i].values
            ma = np.mean(window_prices)
            ma_ratio = ma / self.initial_prices[i]
            ma_ratios.append(ma_ratio)
            if len(window_prices) > 1:
                log_returns = np.diff(np.log(window_prices))
                vol_daily = np.std(log_returns)
                vol_annual = vol_daily * np.sqrt(252)
                avg_ret = np.mean(log_returns)
            else:
                vol_annual = 0.0
                avg_ret = 0.0
            volatilities.append(vol_annual)
            avg_returns.append(avg_ret)
            S = current_prices[i]
            K = S
            if vol_annual > 0:
                d1 = (np.log(S/K) + (r + 0.5*vol_annual**2)*T_const) / (vol_annual*np.sqrt(T_const))
                d2 = d1 - vol_annual*np.sqrt(T_const)
                call_price = S * norm.cdf(d1) - K * np.exp(-r*T_const) * norm.cdf(d2)
            else:
                call_price = S
            bs_ratio = call_price / S
            bs_ratios.append(bs_ratio)
        technical_features = np.concatenate([price_ratio, np.array(ma_ratios),
                                               np.array(volatilities), np.array(bs_ratios),
                                               np.array(avg_returns)])
        state = np.concatenate([portfolio_frac, technical_features])
        return state

    def get_total_value(self, prices=None):
        """
        총 자산 가치 계산: 현금 + 각 주식 보유 가치 (prices는 [종가 혹은 해당일 가격] 배열)
        """
        if prices is None:
            if self.current_day < self.num_days:
                prices = self.price_data.iloc[self.current_day].values
            else:
                prices = self.price_data.iloc[-1].values
        stock_value = sum(self.portfolio[i] * prices[i - 1] for i in range(1, self.num_assets))
        return self.portfolio[0] + stock_value

    def _calculate_relative_reward(self, prev_value, new_value, current_prices):
        """
        보상 체계 개선:  
        - 원시 보상: portfolio 가치 변화  
        - 정규화: 이전 가치 대비 변화량  
        - 벤치마크: 당일 시작 가격과 현재 가격의 평균 비율 변화  
        - 상대 보상: 정규화 보상에서 벤치마크 리턴 차감
        최종 보상은 -1 ~ 1로 클리핑됨.
        """
        raw_reward = new_value - prev_value
        normalized_reward = raw_reward / (prev_value + 1e-6)
        benchmark_returns = (current_prices - self.day_start_prices) / (self.day_start_prices + 1e-6)
        benchmark_return = np.mean(benchmark_returns)
        relative_reward = normalized_reward - benchmark_return
        relative_reward = np.clip(relative_reward, -1.0, 1.0)
        return relative_reward

    def get_portfolio_weights(self, prices):
        """
        현재 포트폴리오의 자산 배분 비율(현금 및 각 주식)에 대한 가중치를 계산합니다.
        prices: 거래가 이루어진 날의 price array (현금은 가격 개념이 없으므로 단순 비율 계산)
        """
        stock_values = np.array([self.portfolio[i] * prices[i - 1] for i in range(1, self.num_assets)])
        cash_value = self.portfolio[0]
        total_value = cash_value + np.sum(stock_values)
        cash_weight = cash_value / total_value
        stock_weights = stock_values / total_value
        return np.concatenate(([cash_weight], stock_weights))

    def step(self, action):
        current_prices = self.price_data.iloc[self.current_day].values  
        prev_value = self.get_total_value(current_prices)
        
        # 이전 포트폴리오 위험 (포트폴리오 가중치의 표준편차)
        prev_weights = self.get_portfolio_weights(current_prices)
        prev_risk = np.std(prev_weights)
        
        # 하루 최대 거래 건수 초과 시 idle 처리
        if action != 0 and len(self.trades_today) >= self.max_trades_per_day:
            action = 0

        trade_reward = 0.0  # 개별 거래 보상 (실현 수익률 등)
        trade_executed = False

        if self.actions[action] == "idle":
            end_day = True
        else:
            end_day = False
            trade = self.actions[action]
            src, dst, frac = trade
            if src == 0 and dst > 0:  # 매수
                available_cash = self.portfolio[0]
                total_trade_amount = frac * available_cash
                effective_trade_amount = total_trade_amount / (1 + self.transaction_cost)
                if effective_trade_amount < 0.1 * current_prices[dst - 1]:
                    # 거래 조건 미달 시 거래 취소
                    pass
                else:
                    shares_bought = effective_trade_amount / current_prices[dst - 1]
                    # 이전 보유량과의 가중평균으로 매입 단가 업데이트
                    old_shares = self.portfolio[dst]
                    old_cost_basis = self.cost_basis[dst]
                    if old_shares > 0:
                        new_total_shares = old_shares + shares_bought
                        new_cost = (old_shares * old_cost_basis + shares_bought * current_prices[dst - 1]) / new_total_shares
                    else:
                        new_cost = current_prices[dst - 1]
                    self.cost_basis[dst] = new_cost

                    self.portfolio[0] -= total_trade_amount
                    self.portfolio[dst] += shares_bought
                    trade_info = {
                        'day': self.current_day,
                        'trade_type': 'buy',
                        'ticker': self.tickers[dst - 1],
                        'dollar_amt': total_trade_amount,
                        'trade_price': current_prices[dst - 1],
                        'shares': shares_bought
                    }
                    self.trades_today.append(trade_info)
                    self.transaction_log.append(trade_info)
                    trade_executed = True
            elif src > 0 and dst == 0:  # 매도
                available_shares = self.portfolio[src]
                total_stock_value = available_shares * current_prices[src - 1]
                transfer_amt = frac * total_stock_value
                if transfer_amt < 0.1 * current_prices[src - 1]:
                    pass
                else:
                    shares_to_sell = transfer_amt / current_prices[src - 1]
                    self.portfolio[src] -= shares_to_sell
                    cash_received = transfer_amt * (1 - self.transaction_cost)
                    self.portfolio[0] += cash_received
                    # 매도 거래에 대해 평균 매입 단가 대비 실제 수익률 (실현 수익) 계산
                    if self.cost_basis[src] > 0:
                        trade_return = (current_prices[src - 1] - self.cost_basis[src]) / self.cost_basis[src]
                        trade_reward += trade_return
                    trade_info = {
                        'day': self.current_day,
                        'trade_type': 'sell',
                        'ticker': self.tickers[src - 1],
                        'dollar_amt': transfer_amt,
                        'trade_price': current_prices[src - 1],
                        'shares': shares_to_sell
                    }
                    self.trades_today.append(trade_info)
                    self.transaction_log.append(trade_info)
                    trade_executed = True
                    # 모든 주식을 매도했으면 매입 단가 초기화
                    if self.portfolio[src] <= 1e-6:
                        self.cost_basis[src] = 0.0

        new_value = self.get_total_value(current_prices)
        base_reward = self._calculate_relative_reward(prev_value, new_value, current_prices)
        if trade_executed:
            base_reward -= self.trade_penalty

        # 기존 위험 보상 (포트폴리오 할당 변화에 따른 보상)
        new_weights = self.get_portfolio_weights(current_prices)
        new_risk = np.std(new_weights)
        risk_bonus = self.risk_reward_weight * (prev_risk - new_risk)
        
        # [추가] 주식 간 상관관계에 따른 분산(헷지) 보너스  
        diversification_bonus = 0.0
        active_stock_indices = [i for i in range(1, self.num_assets) if self.portfolio[i] * current_prices[i - 1] > 0.1]
        if len(active_stock_indices) >= 2 and self.current_day >= self.window:
            returns = []
            for i in active_stock_indices:
                price_series = self.price_data.iloc[self.current_day - self.window + 1 : self.current_day + 1, i - 1].values
                ret = np.diff(price_series) / price_series[:-1]
                returns.append(ret)
            returns = np.array(returns)
            if returns.shape[0] > 1:
                corr_matrix = np.corrcoef(returns)
                off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                avg_corr = np.mean(off_diag)
                diversification_bonus = self.diversification_bonus_weight * (-avg_corr)

        total_reward = base_reward + risk_bonus + trade_reward + diversification_bonus

        # 하루 종료 시(Idle 액션) 추가 보상: Sharpe 비율 및 장기 성장 보너스 적용
        if end_day:
            # Sharpe 비율 기반 보너스 계산
            day_return = (new_value - self.day_start_value) / (self.day_start_value + 1e-6)
            self.daily_returns.append(day_return)
            risk_free_rate = 0.0  # 위험 무시 시 0 또는 지정 값 사용 가능
            returns_array = np.array(self.daily_returns)
            if len(returns_array) > 1:
                avg_ret = np.mean(returns_array)
                std_ret = np.std(returns_array) + 1e-6
                sharpe_ratio = (avg_ret - risk_free_rate) / std_ret
            else:
                sharpe_ratio = 0.0
            sharpe_bonus = self.sharpe_bonus_weight * sharpe_ratio

            # 장기 성장 보너스: 과거 holding_period일과 비교하여 장기 수익 평가
            long_term_bonus = 0.0
            if self.current_day >= self.holding_period:
                past_value = self.daily_history[self.current_day - self.holding_period]
                long_term_return = (new_value - past_value) / (past_value + 1e-6)
                long_term_bonus = self.long_term_bonus_weight * long_term_return

            total_reward += (sharpe_bonus + long_term_bonus)

            # 하루 종료 후 다음 날로 전환
            self.current_day += 1
            self.trades_today = []
            if self.current_day < self.num_days:
                self.day_start_prices = self.price_data.iloc[self.current_day].values.copy()
                self.day_start_value = self.get_total_value(self.day_start_prices)

        self.daily_values.append(new_value)
        self.error_bars.append(np.std(self.portfolio))
        self.daily_history.append(new_value)

        next_state = self._get_state()
        done = (self.current_day >= self.num_days)
        return next_state, total_reward, done, {'portfolio_value': new_value}

########################################
# 3. Dueling DQN 및 DQN 에이전트 (Dropout 및 Randomness Scale 포함)
########################################
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class DQNAgent:
    def __init__(self, state_size, action_size, device, lr=0.0005, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, batch_size=32, target_update=10, randomness_scale=1):
        """
        randomness_scale: 1 ~ 10, 값이 클수록 랜덤성이 높음.
        상태에 가우시안 노이즈를 추가하여 랜덤성을 부여함.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.learn_step_counter = 0
        self.randomness_scale = randomness_scale
        
        self.model = DuelingDQN(state_size, action_size).to(device)
        self.target_model = DuelingDQN(state_size, action_size).to(device)
        self.update_target_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction='none')
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        noise_std = 0.05 * (self.randomness_scale / 10.0)
        noisy_state = state + np.random.normal(0, noise_std, size=state.shape)
        effective_epsilon = self.epsilon * (self.randomness_scale / 10.0)
        if np.random.rand() <= effective_epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(noisy_state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
            q_noise = torch.normal(0, 0.1, size=q_values.shape).to(self.device)
            q_values = q_values + q_noise * (self.randomness_scale / 10.0)
        self.model.train()
        return torch.argmax(q_values[0]).item()
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.model(states).gather(1, actions).squeeze(1)
        next_actions = torch.argmax(self.model(next_states), dim=1, keepdim=True)
        next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        errors = q_values - targets.detach()
        weights = torch.where(rewards < -5, torch.tensor(2.0, device=self.device), torch.tensor(1.0, device=self.device))
        loss = (weights * (errors ** 2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.update_target_network()

########################################
# 4. Main Training 및 Validation Loop (Replay 호출 제한 포함)
########################################
weights_path = "dqn_weights.pth"
train_log_csv = "training_trades.csv"
val_log_csv = "validation_trades.csv"
max_replay_calls_per_episode = 20  # 에피소드 당 최대 replay 호출 횟수
replay_call_interval = 10         # 매 10 스텝마다 replay 호출

if __name__ == "__main__":
    # Training 환경 생성
    train_env = ImprovedStockTradingEnv(train_data, tickers, initial_capital=10000, transaction_cost=0.01, 
                                          window=5, max_trades_per_day=10, cash_fraction=0.1, trade_penalty=0.01)
    state_size = len(train_env._get_state())
    action_size = train_env.action_size
    # Training 시 randomness_scale = 3 (약 30% 정도 랜덤)
    agent = DQNAgent(state_size, action_size, device, randomness_scale=3)
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        agent.model.load_state_dict(state_dict)
        agent.target_model.load_state_dict(agent.model.state_dict())
        print("Pre-trained weights loaded.")
    
    episodes = 10  # 테스트용 에피소드 수
    episode_values = []
    training_trade_logs = []
    
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    
    for e in range(episodes):
        state = train_env.reset()
        done = False
        total_reward = 0
        step_count = 0
        replay_calls = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1
            if step_count % replay_call_interval == 0 and replay_calls < max_replay_calls_per_episode:
                agent.replay()
                replay_calls += 1
        final_value = train_env.daily_history[-1]
        episode_values.append(final_value)
        print(f"Training Episode {e+1}/{episodes} - Final Asset: ${final_value:.2f}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        for tx in train_env.transaction_log:
            tx['episode'] = e+1
            training_trade_logs.append(tx)
        
        torch.save(agent.model.state_dict(), weights_path)
        pd.DataFrame(training_trade_logs).to_csv(train_log_csv, index=False)
        
        ax.clear()
        ax.errorbar(range(1, len(train_env.daily_history)+1), train_env.daily_history, yerr=train_env.error_bars, fmt='o-', capsize=4)
        ax.set_title("Training: Daily Portfolio Value")
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("Portfolio Value ($)")
        plt.pause(0.05)
    
    plt.ioff()
    plt.show()
    print("Saved trained weights and training trade log.")
    
    # Validation Phase (Replay 호출 없음)
    print("\nValidation Phase:")
    val_env = ImprovedStockTradingEnv(val_data, tickers, initial_capital=10000, transaction_cost=0.01, window=5, 
                                        max_trades_per_day=10, cash_fraction=0.1, trade_penalty=0.01)
    agent.randomness_scale = 7  # Validation 시 약 70% 정도 랜덤
    val_episode_values = []
    validation_trade_logs = []
    val_episodes = 5
    
    for e in range(val_episodes):
        state = val_env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = val_env.step(action)
            state = next_state
            total_reward += reward
        final_value = val_env.daily_history[-1]
        val_episode_values.append(final_value)
        print(f"Validation Episode {e+1} - Final Asset: ${final_value:.2f}, Total Reward: {total_reward:.2f}")
        for tx in val_env.transaction_log:
            tx['episode'] = e+1
            validation_trade_logs.append(tx)
        
        plt.figure(figsize=(8,4))
        plt.errorbar(range(1, len(val_env.daily_history)+1), val_env.daily_history, yerr=val_env.error_bars, fmt='o-', capsize=4)
        plt.title(f"Validation Episode {e+1} - Daily Portfolio Value")
        plt.xlabel("Trading Day")
        plt.ylabel("Portfolio Value ($)")
        plt.show()
    
    pd.DataFrame(validation_trade_logs).to_csv(val_log_csv, index=False)
    print(f"Validation trade log saved to {val_log_csv}")

# %%
