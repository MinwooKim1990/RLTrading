# Reinforcement Learning for Stock Trading

## Overview
A deep reinforcement learning trading system implementing Dueling Double Deep Q-Network (DQN) architecture for automated stock trading. The system optimizes trading decisions across multiple stocks while managing risks and transaction costs.

## System Architecture

### 1. Data Module
| Component | Description |
|-----------|-------------|
| Data Source | Yahoo Finance API |
| Asset Coverage | 20 diverse stocks (Tech, Finance, Consumer) |
| Data Processing | Preprocessing and train/validation splitting |
| Time Frame | 6 months of historical data |

### 2. Environment Module
#### State Space Components
| Component | Description | Purpose |
|-----------|-------------|----------|
| Portfolio Allocation | Current asset weights | Track position distribution |
| Price Ratios | Current vs. initial prices | Monitor price movements |
| Moving Averages | Historical price trends | Technical analysis |
| Volatility | Annualized metrics | Risk assessment |
| Black-Scholes Ratio | Option-derived indicators | Value estimation |
| Average Returns | Log return means | Performance tracking |

#### Action Space
| Action Type | Description | Available Options |
|------------|-------------|-------------------|
| Hold | Maintain positions | N/A |
| Buy | Purchase stocks | 25%, 50%, 75%, 100% of cash |
| Sell | Liquidate positions | 25%, 50%, 75%, 100% of holdings |

#### Reward Structure
| Component | Description | Impact |
|-----------|-------------|---------|
| Portfolio Value Change | Daily P&L | Primary reward signal |
| Transaction Costs | Trading fees | Penalty on trades |
| Risk Penalty | Drawdown-based | Risk management |

### 3. Agent Module (Dueling DQN)
#### Network Architecture
| Component | Purpose | Structure |
|-----------|---------|-----------|
| Value Stream | State value estimation | FC layers → Single value |
| Advantage Stream | Action advantage calculation | FC layers → Action values |
| Combination | Q-value computation | V(s) + (A(s,a) - mean(A(s,·))) |

### 4. Training Module
#### Process Components
| Component | Description | Frequency |
|-----------|-------------|-----------|
| Experience Collection | State-action-reward storage | Every step |
| Replay Learning | Batch updates from memory | Configurable interval |
| Target Sync | Network weight updates | Fixed interval |

## Configuration Parameters

### Learning Parameters
| Parameter | Description | Default | Range |
|-----------|-------------|---------|--------|
| Learning Rate | Weight update speed | 0.0005 | 0.0001 - 0.001 |
| Gamma | Future reward discount | 0.99 | 0.95 - 0.99 |
| Epsilon Initial | Starting exploration rate | 1.0 | Fixed |
| Epsilon Minimum | Final exploration rate | 0.01 | 0.01 - 0.1 |
| Epsilon Decay | Exploration reduction rate | 0.995 | 0.99 - 0.999 |
| Batch Size | Training sample count | 32 | 16 - 128 |
| Target Update | Network sync frequency | 10 steps | 5 - 20 |
| Randomness Scale | Exploration factor | 3 (train) / 7 (val) | 1 - 10 |

### Trading Parameters
| Parameter | Description | Default | Range |
|-----------|-------------|---------|--------|
| Initial Capital | Starting funds | $10,000 | Configurable |
| Transaction Cost | Trading fee | 1% | 0.1% - 2% |
| Window Length | Technical indicator period | 5 days | 3 - 20 |
| Risk Penalty | Drawdown coefficient | 0 | 0 - 1 |
| Max Daily Trades | Trading frequency limit | 10 | 5 - 20 |
| Cash Fraction | Initial cash ratio | 10% | 5% - 20% |

## Performance Evaluation

### Return Metrics
| Metric | Description | Calculation |
|--------|-------------|-------------|
| Total Return | Overall performance | (Final Value - Initial) / Initial |
| Daily Returns | Day-to-day changes | Daily value changes |
| Sharpe Ratio | Risk-adjusted return | (Return - Risk Free) / Volatility |

### Risk Metrics
| Metric | Description | Purpose |
|--------|-------------|----------|
| Max Drawdown | Largest peak-to-trough | Risk magnitude |
| Volatility | Return standard deviation | Risk variability |
| Beta | Market sensitivity | Systematic risk |

### Trading Metrics
| Metric | Description | Insight |
|--------|-------------|----------|
| Trade Frequency | Average trades per day | Activity level |
| Trade Size | Mean transaction value | Position sizing |
| Win Rate | Profitable trade ratio | Strategy effectiveness |

## Usage Guide

### Installation
```bash
git clone [repository-url]
cd RLTrading
pip install -r requirements.txt
```

### Operation Modes
1. **Training Mode**
   - Full training cycle
   - Progress visualization
   - Model checkpointing

2. **Validation Mode**
   - Performance evaluation
   - Strategy analysis
   - Risk assessment

3. **Prediction Mode**
   - 10-day forecasting
   - Trade recommendations
   - Portfolio projections

## Current Limitations

### Technical Constraints
- No market impact modeling
- Simplified transaction costs
- Limited technical indicators

### Implementation Gaps
- Basic risk management
- No market regime detection
- Limited asset universe

## Future Enhancements

### Planned Improvements
1. **Risk Management**
   - Advanced portfolio optimization
   - Dynamic risk adjustment
   - Multi-factor risk models

2. **Market Microstructure**
   - Order book integration
   - Liquidity modeling
   - Impact cost estimation

3. **Feature Enhancement**
   - Additional technical indicators
   - Sentiment analysis
   - Market regime detection

## References

1. Mnih, V. et al. (2015) "Human-level control through deep reinforcement learning"
2. Wang et al. (2016) "Dueling Network Architectures for Deep Reinforcement Learning"
3. Van Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-learning" 