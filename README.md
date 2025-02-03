# Reinforcement Learning for Stock Trading

This project implements a Deep Reinforcement Learning (DRL) based trading system using Double DQN (Deep Q-Network) with Dueling architecture. The agent learns to make trading decisions across multiple stocks while managing risk and transaction costs.

## Architecture Overview

### 1. Data Module (`data_module.py`)
- Downloads historical stock data using `yfinance`
- Handles data preprocessing and train/validation splitting
- Provides a default list of 19 diverse stocks including tech, finance, and consumer goods

### 2. Environment Module (`env_module.py`)
- Implements a custom OpenAI Gym-like environment
- **State Space**: Combines multiple features:
  - Portfolio allocation ratios
  - Price ratios relative to initial prices
  - Moving averages
  - Annualized volatilities
  - Black-Scholes option ratios
  - Average log returns
- **Action Space**: Discrete actions including:
  - No action (hold)
  - Buy stocks with different proportions (25%, 50%, 75%, 100%)
  - Sell stocks with different proportions (25%, 50%, 75%, 100%)
- **Reward Function**: Combines multiple factors:
  - Portfolio value change
  - Risk penalty based on maximum drawdown
  - Trading cost penalty
  - Black-Scholes prediction penalty (in validation mode)

### 3. Agent Module (`agent_module.py`)
- Implements Double DQN with Dueling Architecture
- **Network Architecture**:
  - Value Stream: Estimates state value
  - Advantage Stream: Estimates action advantages
  - Combined using V(s) + (A(s,a) - mean(A(s,a)))
- Features:
  - Experience Replay with 10,000 memory size
  - Epsilon-greedy exploration
  - Target network for stable learning
  - Dropout layers for regularization

### 4. Training Module (`train_module.py`)
- Manages the training and validation process
- Provides real-time visualization of:
  - Daily portfolio values
  - Episode returns
  - Training metrics

## Algorithm Process

1. **Initialization**:
   ```
   Initialize DQN with random weights θ
   Initialize target network with weights θ' = θ
   Initialize replay memory D
   ```

2. **Training Loop**:
   ```
   For each episode:
       Reset environment (initial portfolio)
       For each trading day:
           With probability ε select random action
           Otherwise select action a = argmax_a Q(s,a;θ)
           Execute action, observe reward and next state
           Store transition (s,a,r,s') in D
           Sample random minibatch from D
           Set target y = r + γ max_a' Q(s',a';θ') (Double Q-learning)
           Perform gradient descent on (y - Q(s,a;θ))²
           Every C steps reset θ' = θ
   ```

3. **State Processing**:
   ```
   For each stock:
       Calculate price ratios
       Compute moving averages
       Calculate volatility
       Estimate Black-Scholes ratio
       Compute log returns
   Combine with portfolio allocations
   ```

4. **Action Execution**:
   ```
   If action != hold:
       Calculate transfer amount
       Apply transaction cost
       Update portfolio allocation
   Update portfolio values based on price changes
   Calculate reward including penalties
   ```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd RLTrading
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Download historical stock data
2. Initialize the trading environment
3. Train the DQN agent
4. Validate the trained agent
5. Display performance metrics and visualizations

## Requirements
- Python 3.7+
- PyTorch >= 1.9.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- yfinance >= 0.1.63
- matplotlib >= 3.4.0
- scipy >= 1.7.0

## Performance Metrics

The system evaluates performance using:
- Final portfolio value
- Total reward (including penalties)
- Maximum drawdown
- Trading frequency
- Black-Scholes prediction accuracy (in validation)

## Limitations and Future Work

1. **Market Impact**: The current model doesn't account for market impact of trades
2. **Transaction Costs**: Uses a simplified transaction cost model
3. **Feature Engineering**: Could incorporate more technical indicators
4. **Risk Management**: Could implement more sophisticated risk measures
5. **Multi-Agent Systems**: Could explore competitive or cooperative trading agents

## References

1. Mnih, V. et al. (2015) "Human-level control through deep reinforcement learning"
2. Wang et al. (2016) "Dueling Network Architectures for Deep Reinforcement Learning"
3. Van Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-learning" 