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
# 티커 목록: 20개 종목 (기존 19개에 "GOOGL" 추가)
tickers = [
    "AAPL", "DLB", "DIS", "MSFT", "META", "BRK-A", "AVGO", "AMZN", "IONQ", "NVDA",
    "OKLO", "LUNR", "JPM", "CAKE", "KO", "TSLA", "TTWO", "HON", "ARM", "GOOGL"
]

# Download adjusted close prices for the past 6 months
data = yf.download(tickers, period="6mo", interval="1d")["Adj Close"]
data = data.sort_index().dropna()

# Split data: 90% for training, 10% for validation
split_idx = int(len(data) * 0.9)
train_data = data.iloc[:split_idx].reset_index(drop=True)
val_data   = data.iloc[split_idx:].reset_index(drop=True)

########################################
# 2. Improved Stock Trading Environment with Quant Features, Trade Limit & Enhanced Logging
########################################
class ImprovedStockTradingEnv:
    def __init__(self, price_data, tickers, initial_capital=10000, transaction_cost=0.01, window=5, 
                 risk_penalty=0, trade_penalty=1, validation_mode=False, prediction_penalty=50,
                 max_trades_per_day=10, cash_fraction=0.1):
        """
        price_data: DataFrame with each column representing a stock (order corresponds to tickers)
        tickers: list of tickers corresponding to the columns in price_data
        initial_capital: total starting capital
        transaction_cost: cost per trade (e.g., 0.01 means 1% fee)
        window: lookback window for computing technical indicators
        risk_penalty: (여기서는 제거하여, 기본 보상은 자산 변화로 결정)
        trade_penalty: penalty per trade to discourage overtrading
        validation_mode: if True, extra penalty based on BS prediction error is applied
        prediction_penalty: weight for BS prediction error penalty during validation
        max_trades_per_day: maximum number of trades allowed per day
        cash_fraction: fraction of capital to keep as cash initially (here 0.1 → 1,000달러 현금, 9,000달러 투자)
        """
        self.price_data = price_data.copy().reset_index(drop=True)
        self.num_days = len(self.price_data)
        self.num_stocks = self.price_data.shape[1]
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.window = window
        self.risk_penalty = risk_penalty
        self.trade_penalty = trade_penalty
        self.validation_mode = validation_mode
        self.prediction_penalty = prediction_penalty
        self.max_trades_per_day = max_trades_per_day
        self.cash_fraction = cash_fraction
        
        # Portfolio: index 0 = cash; indices 1..num_stocks = dollar value invested in each stock.
        self.num_assets = self.num_stocks + 1
        self.initial_prices = self.price_data.iloc[0].values  
        self.actions = self._create_action_mapping()
        self.action_space = list(self.actions.keys())
        self.action_size = len(self.actions)
        self.portfolio_history = []   # daily portfolio values
        self.transaction_log = []     # global transaction log (list of dicts)
        self.max_portfolio_value = initial_capital
        self.last_bs_ratio = None
        self.trades_today = []        # daily trades log (reset each day)
        self.reset()
        
    def _create_action_mapping(self):
        actions = {0: "end_day"}  # Action 0 means "End Day"
        action_id = 1
        fractions = [0.25, 0.5, 0.75, 1.0]
        # Actions: from cash (index 0) to stock (indices 1..num_stocks)
        for stock in range(1, self.num_assets):
            for frac in fractions:
                actions[action_id] = (0, stock, frac)
                action_id += 1
        # Actions: from stock to cash
        for stock in range(1, self.num_assets):
            for frac in fractions:
                actions[action_id] = (stock, 0, frac)
                action_id += 1
        return actions

    def reset(self):
        self.current_day = 0
        self.portfolio = np.zeros(self.num_assets)
        # 초기 배분: cash_fraction는 현금, 나머지는 각 종목에 균등 투자
        self.portfolio[0] = self.initial_capital * self.cash_fraction
        stock_capital = self.initial_capital * (1 - self.cash_fraction) / self.num_stocks
        for i in range(1, self.num_assets):
            self.portfolio[i] = stock_capital
        self.portfolio_history = [self.get_total_value(self.current_day)]
        self.transaction_log = []
        self.trades_today = []
        self.max_portfolio_value = self.get_total_value(self.current_day)
        state = self._get_state()
        return state

    def _get_state(self):
        total_value = self.get_total_value(self.current_day)
        portfolio_frac = self.portfolio / total_value
        
        if self.current_day < self.num_days:
            current_prices = self.price_data.iloc[self.current_day].values
        else:
            current_prices = self.price_data.iloc[-1].values
        price_ratio = current_prices / self.initial_prices
        
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
        
        self.last_bs_ratio = np.array(bs_ratios)
        technical_features = np.concatenate([price_ratio, np.array(ma_ratios),
                                               np.array(volatilities), np.array(bs_ratios),
                                               np.array(avg_returns)])
        state = np.concatenate([portfolio_frac, technical_features])
        return state

    def get_total_value(self, day_index):
        total = self.portfolio[0]
        if self.current_day < self.num_days:
            current_prices = self.price_data.iloc[self.current_day].values
        else:
            current_prices = self.price_data.iloc[-1].values
        total += np.sum(self.portfolio[1:])
        return total

    def step(self, action):
        """
        - If action == 0 ("end_day"):  
             * Compute bonus for all trades executed today based on next day's prices  
             * Update portfolio value (reward = (new_value - old_value) + bonus_total)  
             * Record summary (티커별 누적 매수/매도) in transaction log  
             * Advance to next day  
        - Else:  
             * Execute trade (if 자산 잔액 충분하면) and record trade details (티커, 거래 유형, 거래 금액, 거래 가격)  
             * Remain on same day to allow multiple trades  
        """
        current_prices = self.price_data.iloc[self.current_day].values  
        value_before = self.get_total_value(self.current_day)
        
        # 하루 거래 횟수 제한: 이미 max_trades_per_day이면 강제로 End Day 처리
        if action != 0 and len(self.trades_today) >= self.max_trades_per_day:
            action = 0
        
        # End Day branch
        if action == 0:
            bonus_total = 0.0
            buy_summary = {}   # {ticker: 누적 매수 금액}
            sell_summary = {}  # {ticker: 누적 매도 금액}
            if self.current_day + 1 < self.num_days:
                next_prices = self.price_data.iloc[self.current_day + 1].values
                for trade_info in self.trades_today:
                    trade_type = trade_info['trade_type']
                    ticker = trade_info['ticker']
                    trade_price = trade_info['trade_price']
                    transfer_amt = trade_info['transfer_amt']
                    ticker_index = self.tickers.index(ticker)
                    new_price = next_prices[ticker_index]
                    if trade_type == "buy":
                        bonus = (new_price / trade_price - 1) * transfer_amt
                        buy_summary[ticker] = buy_summary.get(ticker, 0) + transfer_amt
                    elif trade_type == "sell":
                        bonus = (1 - new_price / trade_price) * transfer_amt
                        sell_summary[ticker] = sell_summary.get(ticker, 0) + transfer_amt
                    else:
                        bonus = 0.0
                    trade_info['bonus'] = bonus
                    bonus_total += bonus
            else:
                bonus_total = 0.0
                buy_summary = {}
                sell_summary = {}
            self.current_day += 1
            if self.current_day >= self.num_days:
                done = True
                next_state = self._get_state()
                final_value = self.get_total_value(self.num_days - 1)
                reward = (final_value - value_before) + bonus_total
                self.portfolio_history.append(final_value)
                info = {'portfolio_value': final_value}
                summary_info = {'day': self.current_day, 'buy_summary': str(buy_summary),
                                'sell_summary': str(sell_summary), 'episode_summary': True}
                self.transaction_log.append(summary_info)
                self.trades_today = []
                return next_state, reward, done, info
            else:
                next_prices = self.price_data.iloc[self.current_day].values
                self.portfolio[1:] = self.portfolio[1:] * (next_prices / current_prices)
                value_after = self.get_total_value(self.current_day)
                reward = (value_after - value_before) + bonus_total  # Total asset change + bonus
                self.portfolio_history.append(value_after)
                done = False
                info = {'portfolio_value': value_after}
                summary_info = {'day': self.current_day, 'buy_summary': str(buy_summary),
                                'sell_summary': str(sell_summary), 'episode_summary': True}
                self.transaction_log.append(summary_info)
                self.trades_today = []
                return self._get_state(), reward, done, info
        
        # Trade action branch
        else:
            trade = self.actions[action]  # (src, dst, fraction)
            src, dst, frac = trade
            available = self.portfolio[src]
            if available <= 0:
                transfer_amt = 0.0
            else:
                transfer_amt = frac * available
            # Execute trade only if transfer_amt > 0
            if transfer_amt > 0:
                self.portfolio[src] -= transfer_amt
                self.portfolio[dst] += transfer_amt * (1 - self.transaction_cost)
            # Determine trade type and ticker
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
            trade_info = {'day': self.current_day,
                          'trade_type': trade_type,
                          'ticker': ticker,
                          'transfer_amt': transfer_amt,
                          'trade_price': trade_price,
                          'bonus': 0.0}
            self.trades_today.append(trade_info)
            self.transaction_log.append(trade_info)
            reward = -self.trade_penalty  # Immediate cost for trading
            current_value = self.get_total_value(self.current_day)
            self.portfolio_history.append(current_value)
            done = False
            info = {'portfolio_value': current_value}
            # Call replay() here for more frequent epsilon decay:
            agent.replay()
            return self._get_state(), reward, done, info

########################################
# 3. Improved Dueling Double DQN Agent with Dropout
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
                 epsilon_min=0.01, epsilon_decay=0.995, batch_size=32, target_update=10):
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
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
        self.model.train()
        return torch.argmax(q_values[0]).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([s for s,_,_,_,_ in minibatch]).to(self.device)
        actions = torch.LongTensor([a for _,a,_,_,_ in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([r for _,_,r,_,_ in minibatch]).to(self.device)
        next_states = torch.FloatTensor([ns for _,_,_,ns,_ in minibatch]).to(self.device)
        dones = torch.FloatTensor([float(d) for _,_,_,_,d in minibatch]).to(self.device)
        
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
# 4. Main Training and Validation Loop with Per-Episode Saving
########################################
weights_path = "dqn_weights.pth"
train_log_csv = "training_trades.csv"
val_log_csv = "validation_trades.csv"

if __name__ == "__main__":
    # Create training environment (90% training data; validation_mode=False)
    train_env = ImprovedStockTradingEnv(train_data, tickers, initial_capital=10000, transaction_cost=0.01, 
                                          window=5, risk_penalty=0, trade_penalty=1,
                                          validation_mode=False, max_trades_per_day=10, cash_fraction=0.1)
    state_size = len(train_env._get_state())
    action_size = train_env.action_size
    agent = DQNAgent(state_size, action_size, device)
    
    if os.path.exists(weights_path):
        agent.model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
        agent.target_model.load_state_dict(agent.model.state_dict())
        print("Loaded pre-trained weights.")
    
    episodes = 100
    episode_values = []
    training_trade_logs = []
    
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    
    for e in range(episodes):
        state = train_env.reset()
        done = False
        daily_values = [train_env.portfolio_history[0]]
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            # Call replay() every step for epsilon decay
            agent.replay()
            state = next_state
            total_reward += reward
            daily_values.append(info['portfolio_value'])
        episode_values.append(daily_values[-1])
        print(f"Training Episode {e+1}/{episodes} - Final Asset: ${daily_values[-1]:.2f}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        for tx in train_env.transaction_log:
            tx['episode'] = e+1
            training_trade_logs.append(tx)
        
        # Save weights and log after each episode
        torch.save(agent.model.state_dict(), weights_path)
        pd.DataFrame(training_trade_logs).to_csv(train_log_csv, index=False)
        
        ax.clear()
        ax.plot(daily_values, marker='o')
        ax.set_title("Training: Daily Portfolio Value")
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("Portfolio Value ($)")
        plt.pause(0.05)
    
    plt.ioff()
    plt.show()
    print("Saved trained weights and training trade log.")
    
    # Validation Phase (10% validation data; validation_mode=True)
    print("\nValidation Phase:")
    val_env = ImprovedStockTradingEnv(val_data, tickers, initial_capital=10000, transaction_cost=0.01, window=5, 
                                        risk_penalty=0, trade_penalty=1,
                                        validation_mode=True, prediction_penalty=50, max_trades_per_day=10, cash_fraction=0.1)
    val_episode_values = []
    validation_trade_logs = []
    val_episodes = 5
    
    for e in range(val_episodes):
        state = val_env.reset()
        done = False
        daily_values = [val_env.portfolio_history[0]]
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = val_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            daily_values.append(info['portfolio_value'])
        val_episode_values.append(daily_values[-1])
        print(f"Validation Episode {e+1} - Final Asset: ${daily_values[-1]:.2f}, Total Reward: {total_reward:.2f}")
        for tx in val_env.transaction_log:
            tx['episode'] = e+1
            validation_trade_logs.append(tx)
        
        plt.figure(figsize=(8,4))
        plt.plot(daily_values, marker='o')
        plt.title(f"Validation Episode {e+1} - Daily Portfolio Value")
        plt.xlabel("Trading Day")
        plt.ylabel("Portfolio Value ($)")
        plt.show()
    
    pd.DataFrame(validation_trade_logs).to_csv(val_log_csv, index=False)
    print(f"Validation trade log saved to {val_log_csv}")


# %%
