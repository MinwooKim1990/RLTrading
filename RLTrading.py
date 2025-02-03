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

########################################
# 1. Data Collection and Preprocessing
########################################
tickers = [
    "AAPL",      # Apple
    "DLB",       # Dolby Laboratories
    "DIS",       # Disney
    "MSFT",      # Microsoft
    "META",      # Meta Platforms
    "BRK-A",     # Berkshire Hathaway A
    "AVGO",      # Broadcom
    "AMZN",      # Amazon
    "IONQ",      # IonQ
    "NVDA",      # Nvidia
    "OKLO",      # Oklo
    "LUNR",      # Intuitive Machines
    "JPM",       # JP Morgan
    "CAKE",      # Cheesecake Factory
    "KO",        # Coca Cola
    "TSLA",      # Tesla
    "TTWO",      # TTWO (assumed as a proxy for T2)
    "HON",       # Honeywell
    "ARM"        # Arm Holdings
]

# Download adjusted close prices for the past 6 months
data = yf.download(tickers, period="1y", interval="1d")["Adj Close"]
data = data.sort_index().dropna()

# Split data: 80% for training and 20% for validation
split_idx = int(len(data) * 0.8)
train_data = data.iloc[:split_idx].reset_index(drop=True)
val_data   = data.iloc[split_idx:].reset_index(drop=True)

########################################
# 2. Improved Stock Trading Environment with Quant Features
########################################
class ImprovedStockTradingEnv:
    def __init__(self, price_data, initial_capital=10000, transaction_cost=0.01, window=5, 
                 risk_penalty=50, trade_penalty=5, validation_mode=False, prediction_penalty=50):
        """
        price_data: pandas DataFrame with each column representing a stock (order follows tickers)
        initial_capital: starting cash
        transaction_cost: cost incurred during transfers (e.g., 0.01 for 1%)
        window: number of days to compute technical indicators
        risk_penalty: penalty applied to drawdown (max portfolio drop)
        trade_penalty: additional penalty when a trade (action != 0) is executed to discourage overtrading
        validation_mode: if True, an extra penalty is applied based on prediction error of BS ratio
        prediction_penalty: weight for prediction error penalty during validation
        """
        self.price_data = price_data.copy().reset_index(drop=True)
        self.num_days = len(self.price_data)
        self.num_stocks = self.price_data.shape[1]
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.window = window
        self.risk_penalty = risk_penalty
        self.trade_penalty = trade_penalty
        self.validation_mode = validation_mode
        self.prediction_penalty = prediction_penalty
        
        # Total assets: index 0 for cash, indices 1..num_stocks for stocks
        self.num_assets = self.num_stocks + 1
        # Baseline prices for ratio computation
        self.initial_prices = self.price_data.iloc[0].values  
        # Create discrete action space (0: do nothing; others: transfer fractions between cash and stocks)
        self.actions = self._create_action_mapping()
        self.action_space = list(self.actions.keys())
        self.action_size = len(self.actions)
        self.portfolio_history = []  # For visualization
        self.max_portfolio_value = initial_capital
        self.last_bs_ratio = None  # To store previous day's BS ratios for validation penalty
        self.reset()
        
    def _create_action_mapping(self):
        actions = {0: None}  # Action 0: do nothing
        action_id = 1
        fractions = [0.25, 0.5, 0.75, 1.0]
        # From cash (index 0) to each stock (indices 1..num_stocks)
        for stock in range(1, self.num_assets):
            for frac in fractions:
                actions[action_id] = (0, stock, frac)
                action_id += 1
        # From stock to cash
        for stock in range(1, self.num_assets):
            for frac in fractions:
                actions[action_id] = (stock, 0, frac)
                action_id += 1
        return actions

    def reset(self):
        self.current_day = 0
        self.portfolio = np.zeros(self.num_assets)
        self.portfolio[0] = self.initial_capital  # Start with all cash
        self.portfolio_history = [self.get_total_value(self.current_day)]
        self.max_portfolio_value = self.initial_capital
        # Initialize last BS ratio from the initial state
        state = self._get_state()
        return state

    def _get_state(self):
        """
        State vector includes:
         - Portfolio fractions (length = num_assets)
         - For each stock (num_stocks), five features:
             1. Price ratio: current price / initial price
             2. Moving average ratio: (moving average over 'window' days) / initial price
             3. Annualized volatility: computed from log returns
             4. BS_ratio: Black-Scholes at-the-money call premium relative to current price
             5. Average log return over the window
        Total state dimension: (num_assets) + (5 * num_stocks)
        """
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
        # For Black-Scholes, use T_const as 30 days in years and a risk-free rate
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
                d1 = (np.log(S/K) + (r + 0.5 * vol_annual**2) * T_const) / (vol_annual * np.sqrt(T_const))
                d2 = d1 - vol_annual * np.sqrt(T_const)
                call_price = S * norm.cdf(d1) - K * np.exp(-r * T_const) * norm.cdf(d2)
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
        """Calculate total portfolio value: cash plus stock assets (in dollar terms)."""
        total = self.portfolio[0]
        if self.current_day < self.num_days:
            current_prices = self.price_data.iloc[self.current_day].values
        else:
            current_prices = self.price_data.iloc[-1].values
        total += np.sum(self.portfolio[1:])
        return total

    def step(self, action):
        """
        1. Execute trade using current day's prices (apply transaction cost and trade penalty if action != 0).
        2. Move to next day and update stock asset values.
        3. Compute reward = (Portfolio change) - (risk penalty * drawdown) - (trade penalty if trade executed)
           and, in validation mode, subtract an extra penalty based on BS_ratio prediction error.
        """
        done = False
        current_prices = self.price_data.iloc[self.current_day].values
        value_before = self.get_total_value(self.current_day)
        
        # Execute trade if action is not 0 (do nothing)
        if action != 0:
            trade = self.actions[action]  # (src, dst, fraction)
            src, dst, frac = trade
            available = self.portfolio[src]
            transfer_amt = frac * available
            self.portfolio[src] -= transfer_amt
            self.portfolio[dst] += transfer_amt * (1 - self.transaction_cost)
            trade_flag = True
        else:
            trade_flag = False
        
        # Save old BS ratios for validation penalty
        old_bs = self.last_bs_ratio.copy() if self.last_bs_ratio is not None else None
        
        # Move to next day
        self.current_day += 1
        if self.current_day >= self.num_days:
            done = True
            next_state = self._get_state()
            final_value = self.get_total_value(self.num_days - 1)
            reward = final_value - value_before
            self.portfolio_history.append(final_value)
            info = {'portfolio_value': final_value}
            return next_state, reward, done, info
        
        # Update portfolio value for stocks by applying price change ratio
        next_prices = self.price_data.iloc[self.current_day].values
        self.portfolio[1:] = self.portfolio[1:] * (next_prices / current_prices)
        
        value_after = self.get_total_value(self.current_day)
        # Update maximum portfolio value and calculate drawdown
        self.max_portfolio_value = max(self.max_portfolio_value, value_after)
        drawdown = (self.max_portfolio_value - value_after) / self.max_portfolio_value
        
        reward = (value_after - value_before) - self.risk_penalty * drawdown
        # Apply trade penalty if a trade was executed
        if trade_flag:
            reward -= self.trade_penalty
        
        next_state = self._get_state()
        
        # In validation mode, add extra penalty based on BS_ratio prediction error
        if self.validation_mode and old_bs is not None:
            # BS ratios are located in the state after portfolio_frac, price_ratio, and ma_ratio, vol, etc.
            start_bs = self.num_assets + 2 * self.num_stocks  # BS ratios start index in technical features
            new_bs = next_state[start_bs : start_bs + self.num_stocks]
            bs_error = np.mean(np.abs(new_bs - old_bs))
            reward -= self.prediction_penalty * bs_error
        
        self.portfolio_history.append(value_after)
        info = {'portfolio_value': value_after}
        return next_state, reward, done, info

########################################
# 3. Improved DQN Agent with Dueling and Double DQN (with Dropout)
########################################
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Advantage stream
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
        self.memory = deque(maxlen=10000)  # Expanded replay memory to 10,000
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update  # update target network every N learning steps
        self.learn_step_counter = 0
        
        self.model = DuelingDQN(state_size, action_size).to(device)
        self.target_model = DuelingDQN(state_size, action_size).to(device)
        self.update_target_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
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
        states = torch.FloatTensor([sample[0] for sample in minibatch]).to(self.device)
        actions = torch.LongTensor([sample[1] for sample in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([sample[2] for sample in minibatch]).to(self.device)
        next_states = torch.FloatTensor([sample[3] for sample in minibatch]).to(self.device)
        dones = torch.FloatTensor([float(sample[4]) for sample in minibatch]).to(self.device)
        
        q_values = self.model(states).gather(1, actions).squeeze(1)
        # Double DQN: use main network for selecting next action, target network for evaluation
        next_actions = torch.argmax(self.model(next_states), dim=1, keepdim=True)
        next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.update_target_network()

########################################
# 4. Main Training and Validation Loop
########################################
if __name__ == "__main__":
    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create training environment (using 80% training data; validation_mode=False)
    train_env = ImprovedStockTradingEnv(train_data, initial_capital=10000, transaction_cost=0.01, 
                                          window=5, risk_penalty=50, trade_penalty=5,
                                          validation_mode=False)
    state_size = len(train_env._get_state())  # e.g., around 115 dimensions (depends on num_stocks and features)
    action_size = train_env.action_size         # discrete action space size
    agent = DQNAgent(state_size, action_size, device)
    
    episodes = 100   # Increase training episodes to 100 to reduce overfitting
    episode_values = []  # Final portfolio value per episode
    
    # Set up real-time visualization (English labels)
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))
    
    for e in range(episodes):
        state = train_env.reset()
        done = False
        daily_values = [train_env.portfolio_history[0]]
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            daily_values.append(info['portfolio_value'])
        
        agent.replay()
        episode_values.append(daily_values[-1])
        print(f"Training Episode {e+1}/{episodes} - Final Asset: ${daily_values[-1]:.2f}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # Plot daily portfolio values for current episode
        ax1.plot(daily_values, label=f"Episode {e+1}")
        ax1.set_title("Training: Daily Portfolio Value")
        ax1.set_xlabel("Trading Day")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(fontsize='small', loc='upper left')
        plt.pause(0.05)
        
        # Plot final portfolio value per episode
        ax2.clear()
        ax2.plot(episode_values, marker='o')
        ax2.set_title("Training: Final Portfolio Value per Episode")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Final Portfolio Value ($)")
        plt.pause(0.05)
    
    plt.ioff()
    plt.show()
    
    # Validation phase (using 20% validation data with validation_mode=True)
    print("\nValidation Phase:")
    val_env = ImprovedStockTradingEnv(val_data, initial_capital=10000, transaction_cost=0.01, window=5, 
                                        risk_penalty=50, trade_penalty=5,
                                        validation_mode=True, prediction_penalty=50)
    val_episode_values = []
    val_episodes = 5  # Number of validation episodes
    
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
        
        plt.figure(figsize=(8,4))
        plt.plot(daily_values, marker='o')
        plt.title(f"Validation Episode {e+1} - Daily Portfolio Value")
        plt.xlabel("Trading Day")
        plt.ylabel("Portfolio Value ($)")
        plt.show()

# %%
