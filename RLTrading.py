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

# -------------------------------
# GPU Acceleration Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Using device:", device)

########################################
# 1. Data Collection and Preprocessing (90% Train, 10% Validation)
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
# 2. Improved Stock Trading Environment
########################################
class ImprovedStockTradingEnv:
    def __init__(self, price_data, tickers, initial_capital=10000, transaction_cost=0.01, window=5, 
                 max_trades_per_day=10, cash_fraction=0.1):
        """
        - initial_capital: total dollars available.
        - cash_fraction: fraction held as cash initially.
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
        
        # Portfolio: index 0 = cash, indices 1..num_stocks = number of shares held.
        self.num_assets = self.num_stocks + 1
        self.initial_prices = self.price_data.iloc[0].values  
        self.actions = self._create_action_mapping()
        self.action_space = list(self.actions.keys())
        self.action_size = len(self.actions)
        # For plotting and logging.
        self.daily_history = []   # End Day portfolio values.
        self.error_bars = []      # Error bars per day.
        self.transaction_log = [] # Global trade log.
        self.trades_today = []    # Trades executed in current day.
        self.daily_values = []    # Intermediate portfolio values for current day.
        self.reset()
        
    def _create_action_mapping(self):
        actions = {0: "end_day"}
        action_id = 1
        fractions = [0.25, 0.5, 0.75, 1.0]
        for stock in range(1, self.num_assets):
            for frac in fractions:
                actions[action_id] = (0, stock, frac)
                action_id += 1
        for stock in range(1, self.num_assets):
            for frac in fractions:
                actions[action_id] = (stock, 0, frac)
                action_id += 1
        return actions

    def reset(self):
        self.current_day = 0
        self.portfolio = np.zeros(self.num_assets)
        self.portfolio[0] = self.initial_capital * self.cash_fraction
        stock_money = self.initial_capital * (1 - self.cash_fraction)
        for i in range(1, self.num_assets):
            self.portfolio[i] = stock_money / self.num_stocks / self.initial_prices[i - 1]
        initial_value = self.get_total_value()
        self.daily_history = [initial_value]
        self.error_bars = [0.0]
        self.transaction_log = []
        self.daily_values = [initial_value]
        self.trades_today = []
        # 추가: 일 시작시 baseline 값 설정 (현금, 각 주식 보유량, 시작 가격)
        self.baseline_cash = self.portfolio[0]
        self.baseline_shares = self.portfolio[1:].copy()  # 각 주식의 baseline 보유 수량
        self.day_start_prices = self.price_data.iloc[self.current_day].values.copy()
        state = self._get_state()
        return state

    def _get_state(self):
        if self.current_day < self.num_days:
            current_prices = self.price_data.iloc[self.current_day].values
        else:
            current_prices = self.price_data.iloc[-1].values
        stock_values = np.array([self.portfolio[i] * current_prices[i - 1] for i in range(1, self.num_assets)])
        total_value = self.portfolio[0] + np.sum(stock_values)
        portfolio_frac = np.concatenate(([self.portfolio[0] / total_value], stock_values / total_value))
        price_ratio = current_prices / self.initial_prices
        
        # Technical features: moving averages, volatilities, Black-Scholes ratios, average returns.
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

    def get_total_value(self):
        if self.current_day < self.num_days:
            current_prices = self.price_data.iloc[self.current_day].values
        else:
            current_prices = self.price_data.iloc[-1].values
        stock_values = np.array([self.portfolio[i] * current_prices[i - 1] for i in range(1, self.num_assets)])
        total = self.portfolio[0] + np.sum(stock_values)
        return total

    def step(self, action):
        # 중간 포트폴리오 가치 기록 (에러바 계산용)
        curr_val = self.get_total_value()
        self.daily_values.append(curr_val)
        
        if self.current_day >= self.num_days:
            return self._get_state(), 0, True, {'portfolio_value': curr_val}
        
        current_prices = self.price_data.iloc[self.current_day].values  
        if action != 0 and len(self.trades_today) >= self.max_trades_per_day:
            action = 0
        # 같은 거래가 당일 3회 이상 발생한 경우, 강제로 end_day 처리 (기존 코드와 동일)
        if action != 0:
            trade = self.actions[action]
            src, dst, frac = trade
            if src == 0:
                trade_type = "buy"
                ticker = self.tickers[dst - 1]
                trade_price = current_prices[dst - 1]
            elif dst == 0:
                trade_type = "sell"
                ticker = self.tickers[src - 1]
                trade_price = current_prices[src - 1]
            else:
                trade_type = "transfer"
                ticker = "N/A"
                trade_price = None
            similar_count = sum(1 for t in self.trades_today if t.get('trade_type') == trade_type 
                                and t.get('ticker') == ticker 
                                and (t.get('trade_price') is not None and abs(t.get('trade_price') - trade_price) < 1e-3))
            if similar_count >= 3:
                action = 0

        # End Day 분기 --------------------------------------------------------------------------------
        if action == 0:
            # 하루 시작시 기준포트폴리오(현금 및 보유 주식)의 총 가치 계산
            prev_total_value = self.baseline_cash + np.sum(self.baseline_shares * self.day_start_prices)
            
            # 기존과 같이 하루 마감 시, 다음 날 가격을 반영하기 위해 current_day를 증가시킵니다.
            self.current_day += 1
            if self.current_day < self.num_days:
                day_end_prices = self.price_data.iloc[self.current_day].values.copy()
            else:
                day_end_prices = self.price_data.iloc[-1].values.copy()
            new_total_value = self.portfolio[0] + np.sum(self.portfolio[1:] * day_end_prices)
            
            # 전체 보상(전체 포트폴리오 가치 변화)
            total_reward = new_total_value - prev_total_value
            
            # 개별 거래에 의한 보상 계산 (수수료 1% 반영)
            trade_reward_total = 0.0
            baseline_shares_updated = self.baseline_shares.copy()
            for trade in self.trades_today:
                ticker = trade.get('ticker')
                i = self.tickers.index(ticker)  # 주식의 인덱스 (티커순)
                if trade.get('trade_type') == 'buy':
                    trade_price = trade.get('trade_price')
                    shares = trade.get('shares')
                    # 매수한 주식은 수수료 1%가 포함된 거래가격 기준으로 보상 계산
                    trade_reward = (day_end_prices[i] - trade_price * (1 + self.transaction_cost)) * shares
                    trade_reward_total += trade_reward
                elif trade.get('trade_type') == 'sell':
                    trade_price = trade.get('trade_price')
                    shares = trade.get('shares')
                    # 매도한 주식은 거래 후 수수료 반영된 금액과 당일 시작 가격과의 차이로 보상 계산
                    trade_reward = (trade_price * (1 - self.transaction_cost) - self.day_start_prices[i]) * shares
                    trade_reward_total += trade_reward
                    baseline_shares_updated[i] = max(baseline_shares_updated[i] - shares, 0)
            
            # 거래가 없었던 주식에 대한 보유 보상: 당일 시작 가격과 종가의 차이
            holdings_reward = 0.0
            for i in range(len(baseline_shares_updated)):
                holdings_reward += baseline_shares_updated[i] * (day_end_prices[i] - self.day_start_prices[i])
            
            # 현금에 대한 보상: 현금 잔고 차이
            cash_reward = self.portfolio[0] - self.baseline_cash
            
            # 최종 보상은 개별 거래 보상 + 보유 보상 + 현금 보상
            reward = trade_reward_total + holdings_reward + cash_reward
            # (디버그 확인: total_reward와 reward는 같아야 합니다.)
            # print(f"total_reward: {total_reward:.2f}, cash:{cash_reward:.2f}, holdings:{holdings_reward:.2f}, trades:{trade_reward_total:.2f}")
            
            if len(self.daily_values) > 1:
                err = (max(self.daily_values) - min(self.daily_values)) / 2.0
            else:
                err = 0.0
            self.daily_history.append(new_total_value)
            self.error_bars.append(err)
            
            summary_info = {'day': self.current_day, 'buy_summary': str({}),
                            'sell_summary': str({}), 'episode_summary': True}
            self.transaction_log.append(summary_info)
            self.trades_today = []
            self.daily_values = []
            
            # 새로운 날 시작 시 baseline 업데이트 (현금, 주식, 일 시작 가격)
            if self.current_day < self.num_days:
                self.baseline_cash = self.portfolio[0]
                self.baseline_shares = self.portfolio[1:].copy()
                self.day_start_prices = day_end_prices.copy()
            
            return self._get_state(), reward, self.current_day >= self.num_days, {'portfolio_value': new_total_value}
        
        # Trade 액션 분기 (거래 실행) --------------------------------------------------------------------
        else:
            trade = self.actions[action]
            src, dst, frac = trade
            if src == 0:  # 매수
                available_cash = self.portfolio[0]
                total_trade_amount = frac * available_cash
                effective_trade_amount = total_trade_amount / (1 + self.transaction_cost)
                if effective_trade_amount < 0.1 * current_prices[dst - 1]:
                    return self._get_state(), 0, False, {'portfolio_value': self.get_total_value()}
                shares_bought = effective_trade_amount / current_prices[dst - 1]
                self.portfolio[0] -= total_trade_amount  
                self.portfolio[dst] += shares_bought
                trade_type = "buy"
                ticker = self.tickers[dst - 1]
                trade_price = current_prices[dst - 1]
                dollar_amt = total_trade_amount
                trade_info = {'day': self.current_day,
                              'trade_type': trade_type,
                              'ticker': ticker,
                              'dollar_amt': dollar_amt,
                              'trade_price': trade_price,
                              'shares': shares_bought}
            elif dst == 0:  # 매도
                available_shares = self.portfolio[src]
                total_stock_value = available_shares * current_prices[src - 1]
                transfer_amt = frac * total_stock_value
                if transfer_amt < 0.1 * current_prices[src - 1]:
                    return self._get_state(), 0, False, {'portfolio_value': self.get_total_value()}
                shares_to_sell = transfer_amt / current_prices[src - 1]
                self.portfolio[src] -= shares_to_sell
                cash_received = transfer_amt * (1 - self.transaction_cost)
                self.portfolio[0] += cash_received
                trade_type = "sell"
                ticker = self.tickers[src - 1]
                trade_price = current_prices[src - 1]
                dollar_amt = transfer_amt
                trade_info = {'day': self.current_day,
                              'trade_type': trade_type,
                              'ticker': ticker,
                              'dollar_amt': dollar_amt,
                              'trade_price': trade_price,
                              'shares': shares_to_sell}
            else:
                return self._get_state(), 0, False, {'portfolio_value': self.get_total_value()}
            self.trades_today.append(trade_info)
            self.transaction_log.append(trade_info)
            reward = 0  # 개별 거래 시 즉시 보상은 없으며, 후속 end_day에서 계산합니다.
            new_val = self.get_total_value()
            self.daily_values.append(new_val)
            return self._get_state(), reward, False, {'portfolio_value': new_val}

########################################
# 3. Dueling Double DQN Agent with Dropout and Randomness Scale
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
        randomness_scale: 1 ~ 10, where 1 means ~10% randomness and 10 means 100% randomness.
        effective_epsilon = epsilon * (randomness_scale / 10)
        상태에 가우시안 노이즈를 추가하여 랜덤성을 부여합니다.
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
# 4. Main Training and Validation Loop with Replay Call Limit
########################################
weights_path = "dqn_weights.pth"
train_log_csv = "training_trades.csv"
val_log_csv = "validation_trades.csv"
max_replay_calls_per_episode = 20  # 최대 replay 호출 횟수 per 에피소드
replay_call_interval = 10         # 매 10 스텝마다 replay 호출

if __name__ == "__main__":
    # Create training environment
    train_env = ImprovedStockTradingEnv(train_data, tickers, initial_capital=10000, transaction_cost=0.01, 
                                          window=5, max_trades_per_day=10, cash_fraction=0.1)
    state_size = len(train_env._get_state())
    action_size = train_env.action_size
    # For training, randomness_scale = 3 (~30% randomness)
    agent = DQNAgent(state_size, action_size, device, randomness_scale=3)
    
    if os.path.exists(weights_path):
        agent.model.load_state_dict(torch.load(weights_path, map_location=device))
        agent.target_model.load_state_dict(agent.model.state_dict())
        print("Loaded pre-trained weights.")
    
    episodes = 10  # For testing, set to 10 episodes
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
            step_count += 1
            if step_count % replay_call_interval == 0 and replay_calls < max_replay_calls_per_episode:
                agent.replay()
                replay_calls += 1
            state = next_state
            total_reward += reward
        # Use daily_history and error_bars for plotting.
        day_values = train_env.daily_history  # one value per trading day
        if len(day_values) != len(train_env.error_bars):
            train_env.error_bars.append(0.0)
        episode_values.append(day_values[-1])
        print(f"Training Episode {e+1}/{episodes} - Final Asset: ${day_values[-1]:.2f}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        for tx in train_env.transaction_log:
            tx['episode'] = e+1
            training_trade_logs.append(tx)
        
        torch.save(agent.model.state_dict(), weights_path)
        pd.DataFrame(training_trade_logs).to_csv(train_log_csv, index=False)
        
        ax.clear()
        ax.errorbar(range(1, len(day_values)+1), day_values, yerr=train_env.error_bars, fmt='o-', capsize=4)
        ax.set_title("Training: End Day Portfolio Value")
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("Portfolio Value ($)")
        plt.pause(0.05)
    
    plt.ioff()
    plt.show()
    print("Saved trained weights and training trade log.")
    
    # Validation Phase (no replay calls)
    print("\nValidation Phase:")
    val_env = ImprovedStockTradingEnv(val_data, tickers, initial_capital=10000, transaction_cost=0.01, window=5, 
                                        max_trades_per_day=10, cash_fraction=0.1)
    agent.randomness_scale = 7  # For validation, ~70% randomness
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
        day_values = val_env.daily_history
        if len(day_values) != len(val_env.error_bars):
            val_env.error_bars.append(0.0)
        val_episode_values.append(day_values[-1])
        print(f"Validation Episode {e+1} - Final Asset: ${day_values[-1]:.2f}, Total Reward: {total_reward:.2f}")
        for tx in val_env.transaction_log:
            tx['episode'] = e+1
            validation_trade_logs.append(tx)
        
        plt.figure(figsize=(8,4))
        plt.errorbar(range(1, len(day_values)+1), day_values, yerr=val_env.error_bars, fmt='o-', capsize=4)
        plt.title(f"Validation Episode {e+1} - End Day Portfolio Value")
        plt.xlabel("Trading Day")
        plt.ylabel("Portfolio Value ($)")
        plt.show()
    
    pd.DataFrame(validation_trade_logs).to_csv(val_log_csv, index=False)
    print(f"Validation trade log saved to {val_log_csv}")
