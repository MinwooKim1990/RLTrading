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

## Detailed Explanation of Key Components

### 1. Black-Scholes Ratio
- **What it is:** The Black-Scholes formula is a mathematical model used to estimate the price of a call option. In this project, the formula computes a call option price for each stock which is then divided by the current stock price to form a Black-Scholes ratio.
- **Why it is used:** This ratio serves as an additional technical indicator, helping the agent gauge potential value from an option pricing perspective. It adds another dimension to the state representation beyond traditional price indicators.

### 2. Actions
- **Definition:** The action space is discrete and consists of:
  - **Action 0:** Hold (do nothing).
  - **Buy Actions:** Transfer a fraction (25%, 50%, 75%, or 100%) of the available cash to purchase a specific stock.
  - **Sell Actions:** Transfer a fraction (25%, 50%, 75%, or 100%) of a stock's current value back to cash.
- **Purpose:** These actions allow the agent to experiment with different trading strategies, balancing risk and reward by choosing to invest more or liquidate assets as market conditions change.

### 3. Reward Function
- **Calculation:** The reward is based on the change in portfolio value from one day to the next.
  - **Risk Penalty:** A deduction proportional to the drawdown (decline from the maximum portfolio value) is applied to discourage excessive risk-taking.
  - **Trading Penalty:** A cost is subtracted for every transaction executed, simulating transaction fees.
  - **Validation Penalty:** In validation mode, an extra penalty is imposed if there is a significant error in the Black-Scholes ratio prediction.
- **Purpose:** The reward function is designed to promote steady portfolio growth while minimizing unnecessary trades and risk exposure.

### 4. Window Parameter
- **Definition:** The window parameter indicates the number of past trading days used when calculating technical indicators such as moving averages, annualized volatility, and average log returns.
- **Purpose:** A larger window provides a broader historical perspective, while a smaller window focuses more on recent trends. This parameter is crucial in balancing between long-term trends and short-term fluctuations in the market.

### 5. Value Stream vs. Advantage Stream (Dueling Network Architecture)
- **Value Stream:** Estimates the overall value of being in a specific state regardless of the action taken.
- **Advantage Stream:** Estimates the benefit (or advantage) of taking a particular action compared to the average of all possible actions in that state.
- **Combined Calculation:** The final Q-value is computed as:
  
  Q(s, a) = Value(s) + (Advantage(s, a) - Average(Advantage(s, ·)))
  
  This design helps the network learn the state's intrinsic value independently from the specifics of each action.

### 6. Experience Replay
- **What it is:** A technique where past experiences (state, action, reward, next state, done) are stored in a buffer.
- **Why it is used:** During training, random batches of experiences are sampled to break the correlation between sequential data. This approach leads to more stable and efficient learning by reusing past experiences multiple times.

### 7. Epsilon-Greedy Exploration
- **What it is:** A strategy for balancing exploration and exploitation.
- **How it works:** With probability epsilon, the agent selects a random action (exploration). With probability (1 - epsilon), it selects the best-known action (exploitation) based on current Q-value estimates.
- **Decay Mechanism:** The epsilon value starts high (e.g., 1.0) to encourage exploration, gradually decays (e.g., multiplied by 0.995 each update) towards a minimum value (e.g., 0.01) to increasingly favor exploitation as the agent learns.

### 8. Double DQN and Target Network
- **Double DQN:** This method decouples the action selection and value evaluation to reduce overestimation of Q-values. The main network selects the best action while the target network evaluates its Q-value.
- **Target Network:** A separate copy of the main network, updated less frequently. It provides a stable reference during training, and its parameters are synchronized with the main network every fixed number of steps (target update interval), such as every 10 steps.

### 9. Discount Factor (Gamma) and Epsilon Parameters
- **Discount Factor (Gamma):** A parameter between 0 and 1 that determines how much future rewards are valued relative to immediate rewards. A higher gamma prioritizes long-term gains.
- **Epsilon Parameters:**
  - **Initial Epsilon:** Starting exploration rate (e.g., 1.0) to ensure ample exploration at the beginning.
  - **Epsilon Decay:** The multiplicative factor (e.g., 0.995) applied after each training step to gradually reduce exploration.
  - **Minimum Epsilon:** A lower bound (e.g., 0.01) to ensure that some degree of exploration always remains.
- **Target Update Interval:** The number of training steps between updates of the target network. This parameter helps maintain stability in training by updating the target network less frequently than the main network.

---

## Korean Translation (한국어 번역)

### 1. 블랙-숄즈 비율
- **정의:** 블랙-숄즈 공식은 콜 옵션의 가격을 추정하기 위해 사용되는 수학적 모델입니다. 이 프로젝트에서는 각 주식에 대해 콜 옵션 가격을 계산한 후, 이를 현재 주가로 나누어 블랙-숄즈 비율을 도출합니다.
- **사용 목적:** 이 비율은 추가적인 기술 지표로 활용되어, 옵션 가격 책정 관점에서 잠재적인 가치를 평가하는 데 도움을 줍니다. 이는 전통적인 가격 지표 외에 상태 표현에 추가적인 정보를 제공합니다.

### 2. 행동
- **정의:** 환경에서의 행동 공간은 이산적(discrete)입니다.
  - **행동 0:** 아무것도 하지 않음 (보유).
  - **매수 행동:** 사용 가능한 현금의 일정 비율(25%, 50%, 75%, 100%)을 주식 구입에 사용.
  - **매도 행동:** 주식의 현재 가치의 일정 비율(25%, 50%, 75%, 100%)을 현금으로 전환.
- **목적:** 이러한 행동은 에이전트가 다양한 거래 전략을 실험할 수 있게 하며, 시장 상황 변화에 따라 위험과 보상을 균형 있게 관리할 수 있도록 돕습니다.

### 3. 보상 함수
- **계산:** 보상은 하루 전과 후의 포트폴리오 가치 변화에 기반하여 계산됩니다.
  - **리스크 패널티:** 최대 손실(최대 포트폴리오 가치 대비 하락폭)에 비례하여 패널티를 부여하여 과도한 위험을 피하도록 유도합니다.
  - **거래 패널티:** 거래가 발생할 때마다 거래 수수료와 유사한 비용이 차감됩니다.
  - **검증 패널티:** 검증 모드에서는 블랙-숄즈 비율 예측 오차가 클 경우 추가 패널티가 부과됩니다.
- **목적:** 이 보상 함수는 지속적인 포트폴리오 성장과 동시에 불필요한 거래와 과도한 위험 노출을 줄이는 것을 목표로 합니다.

### 4. 윈도우 파라미터
- **정의:** 윈도우 파라미터는 이동평균, 연간 변동성, 평균 로그 수익률과 같은 기술 지표를 계산할 때 참조하는 과거 거래일의 수를 의미합니다.
- **목적:** 윈도우 기간이 길면 과거 데이터를 넓게 반영할 수 있고, 짧으면 최근의 시장 동향에 민감하게 반응합니다. 이는 장기 트렌드와 단기 변동성 사이의 균형을 맞추는 데 중요한 역할을 합니다.

### 5. 가치 스트림 vs. 이점 스트림 (듀얼링 네트워크 구조)
- **가치 스트림:** 특정 상태에 있을 때의 전반적인 가치를 추정합니다. 이는 행동 선택과 무관하게 상태 자체의 가치를 평가합니다.
- **이점 스트림:** 특정 행동을 취함으로써 얻을 수 있는 추가적인 이점을 추정합니다. 즉, 평균적인 행동 대비 특정 행동의 부가적인 효과를 계산합니다.
- **결합 방식:** 최종 Q-값은 다음과 같이 계산됩니다:
  
  Q(s, a) = Value(s) + (Advantage(s, a) - Average(Advantage(s, ·)))
  
  이러한 구조는 상태의 내재적 가치와 각 행동의 효과를 보다 효과적으로 학습할 수 있도록 돕습니다.

### 6. 경험 재생
- **정의:** 과거의 경험(상태, 행동, 보상, 다음 상태, 종료 여부)을 버퍼에 저장해 두는 기법입니다.
- **사용 목적:** 학습 시, 저장된 경험들 중 무작위 배치를 샘플링하여 네트워크를 업데이트함으로써, 연속된 데이터 간 상관관계를 줄이고 안정적인 학습을 가능하게 합니다.

### 7. 입실론 그리디 탐색
- **정의:** 탐험과 활용 사이의 균형을 맞추기 위한 전략입니다.
- **작동 방식:** 일정 확률(epsilon)로 무작위 행동을 선택하여 새로운 가능성을 탐색하고, 나머지 (1 - epsilon) 확률로 현재 정책에 따라 가장 좋은 행동을 선택합니다.
- **감쇠 메커니즘:** 초기 높은 epsilon(예: 1.0)에서 시작하여 학습이 진행됨에 따라 epsilon을 점차 감소(예: 0.995로 곱셈)시켜 최소값(예: 0.01)까지 줄어들게 하여, 점진적으로 활용을 우선시합니다.

### 8. 더블 DQN 및 타겟 네트워크
- **더블 DQN:** 행동 선택과 가치 평가를 분리하여 Q-값 과대평가 문제를 해결합니다. 메인 네트워크가 최적의 행동을 선택하고, 타겟 네트워크가 그 행동의 Q-값을 평가합니다.
- **타겟 네트워크:** 메인 네트워크의 복사본으로, 업데이트 간격(예: 매 10 스텝)마다 메인 네트워크의 파라미터로 동기화됩니다. 이는 학습 시 안정적인 참조값을 제공하여 학습을 안정화합니다.

### 9. 할인율 (감마) 및 입실론 파라미터
- **할인율 (Gamma):** 0과 1 사이의 값으로, 미래 보상에 부여하는 가중치를 결정합니다. 높은 gamma 값은 장기적인 이익을 더 중시합니다.
- **입실론 파라미터:**
  - **초기 입실론:** 탐색을 충분히 진행하기 위해 시작 시 높은 값(예: 1.0)을 사용합니다.
  - **입실론 감쇠:** 매 학습 단계마다 일정 비율(예: 0.995)을 곱해 점차 감소시킵니다.
  - **최소 입실론:** 학습 후반에도 소폭의 탐색이 이루어지도록 최소값(예: 0.01)을 유지합니다.
- **타겟 업데이트 간격:** 타겟 네트워크를 메인 네트워크와 동기화하는 빈도(예: 매 10 스텝). 이 값은 학습의 안정성에 중요한 영향을 미칩니다. 