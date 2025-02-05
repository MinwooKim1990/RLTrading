# %%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from modules.data_module import DataModule
from modules.env_module import ImprovedStockTradingEnv
from modules.agent_module import DQNAgent
from modules.train_module import TrainModule

# Configuration Parameters
CONFIG = {
    # Data Parameters
    "tickers": [
        "AAPL", "DLB", "DIS", "MSFT", "META", "BRK-A", "AVGO", "AMZN", "IONQ", "NVDA",
        "OKLO", "LUNR", "JPM", "CAKE", "KO", "TSLA", "TTWO", "HON", "ARM", "GOOGL"
    ],
    "data_period": "6mo",
    "data_interval": "1d",
    "train_split": 0.9,
    
    # Environment Parameters
    "initial_capital": 10000,
    "transaction_cost": 0.01,
    "window": 5,
    "max_trades_per_day": 10,
    "cash_fraction": 0.1,
    
    # Agent Parameters
    "learning_rate": 0.001,
    "gamma": 0.98,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.99,
    "batch_size": 64,
    "target_update": 10,
    "train_randomness": 7,
    "val_randomness": 3,
    
    # Training Parameters
    "train_episodes": 10,
    "val_episodes": 10,
    "max_replay_calls_per_episode": 10,
    "replay_call_interval": 10,
    
    # File Paths
    "weights_path": "dqn_weights_volatility.pth",
    "train_log_csv": "training_trades.csv",
    "val_log_csv": "validation_trades.csv"
}

# GPU Acceleration Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Using device:", device)

def train(config):
    """
    Training function for the trading agent
    학습 에이전트를 위한 훈련 함수
    """
    # Initialize modules
    data_module = DataModule(
        config["tickers"],
        period=config["data_period"],
        interval=config["data_interval"],
        train_split=config["train_split"]
    )
    train_data, val_data = data_module.get_data()
    
    # Setup training environment
    train_env = ImprovedStockTradingEnv(
        train_data, config["tickers"],
        initial_capital=config["initial_capital"],
        transaction_cost=config["transaction_cost"],
        window=config["window"],
        max_trades_per_day=config["max_trades_per_day"],
        cash_fraction=config["cash_fraction"]
    )
    
    # Initialize agent
    state_size = len(train_env._get_state())
    action_size = train_env.action_size
    agent = DQNAgent(
        state_size, action_size, device,
        lr=config["learning_rate"],
        gamma=config["gamma"],
        epsilon=config["epsilon"],
        epsilon_min=config["epsilon_min"],
        epsilon_decay=config["epsilon_decay"],
        batch_size=config["batch_size"],
        target_update=config["target_update"],
        randomness_scale=config["train_randomness"]
    )
    
    # Initialize training module
    train_module = TrainModule(
        train_env=train_env,
        val_env=None,
        agent=agent,
        device=device,
        weights_path=config["weights_path"],
        train_log_csv=config["train_log_csv"],
        val_log_csv=config["val_log_csv"],
        max_replay_calls_per_episode=config["max_replay_calls_per_episode"],
        replay_call_interval=config["replay_call_interval"]
    )
    
    # Start training
    train_module.train(episodes=config["train_episodes"])

def validate(config):
    """
    Validation function with bootstrap analysis
    부트스트랩 분석과 함께, 각 거래일마다 평가된 포트폴리오 가치를 에피소드별로 수집하는 검증 함수
    """
    # Initialize modules
    data_module = DataModule(
        config["tickers"],
        period=config["data_period"],
        interval=config["data_interval"],
        train_split=config["train_split"]
    )
    train_data, val_data = data_module.get_data()
    
    # 검증 데이터 정보 출력
    print("\nValidation Phase Started")
    print("=======================")
    
    # 검증 환경 설정
    val_env = ImprovedStockTradingEnv(
        val_data, config["tickers"],
        initial_capital=config["initial_capital"],
        transaction_cost=config["transaction_cost"],
        window=config["window"],
        max_trades_per_day=config["max_trades_per_day"],
        cash_fraction=config["cash_fraction"]
    )
    
    # 에이전트 초기화
    state_size = len(val_env._get_state())
    action_size = val_env.action_size
    agent = DQNAgent(
        state_size, action_size, device,
        randomness_scale=config["val_randomness"]
    )
    
    # 학습된 가중치 로드
    agent.model.load_state_dict(torch.load(config["weights_path"], map_location=device, weights_only=False))
    agent.target_model.load_state_dict(agent.model.state_dict())
    
    # 에피소드별로 검증을 실행하며, 각 거래일마다의 포트폴리오 가치를 수집 (거래 횟수에 상관없이 하루에 한 번 기록)
    all_portfolio_values = []
    
    print("\nRunning validation episodes...")
    for e in range(config["val_episodes"]):
        state = val_env.reset()
        done = False
        portfolio_values = []
        # 현재 거래일을 기록 (초기 거래일)
        last_recorded_day = val_env.current_day
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = val_env.step(action)
            
            # 거래일이 변경되었으면, 그 날 평가된 포트폴리오 가치를 기록
            if val_env.current_day != last_recorded_day:
                portfolio_values.append(info['portfolio_value'])
                last_recorded_day = val_env.current_day
            
            state = next_state
        
        # 에피소드 종료 후, 마지막 거래일의 포트폴리오 가치가 기록되지 않았다면 추가
        if not portfolio_values or portfolio_values[-1] != info['portfolio_value']:
            portfolio_values.append(info['portfolio_value'])
            
        all_portfolio_values.append(portfolio_values)
        print(f"Episode {e+1}: Final Portfolio Value: ${portfolio_values[-1]:.2f}")
    
    # 각 에피소드의 기록이 실제 거래일 수(val_data의 길이)와 일치하지 않을 경우, 부족한 부분은 마지막 값으로 채워 경고가 뜨지 않도록 함
    max_days = len(val_data)
    for idx, values in enumerate(all_portfolio_values):
        if len(values) < max_days:
            all_portfolio_values[idx] = values + [values[-1]] * (max_days - len(values))
        elif len(values) > max_days:
            all_portfolio_values[idx] = values[:max_days]
            
    # 부트스트랩 분석 그래프 생성
    print("\nGenerating bootstrap analysis...")
    
    plt.figure(figsize=(12, 6))
    
    # 모든 에피소드별로 각 거래일의 포트폴리오 가치를 저장할 배열 초기화
    daily_values = [[] for _ in range(max_days)]
    
    # 에피소드별로 거래일의 포트폴리오 가치를 수집
    for episode_values in all_portfolio_values:
        for day, value in enumerate(episode_values):
            if day < max_days:  # 실제 거래일 수 내의 데이터만 사용
                daily_values[day].append(value)
    
    # 각 거래일의 통계 계산 (평균 및 95% 신뢰구간)
    mean_values = []
    ci_upper = []
    ci_lower = []
    
    for day_values in daily_values:
        # 빈 리스트에 대해서 np.mean, np.std 호출 시 발생하는 경고를 방지하기 위해 조건문 사용
        if day_values:
            mean_val = np.mean(day_values)
            std_val = np.std(day_values)
        else:
            mean_val = np.nan
            std_val = 0
        mean_values.append(mean_val)
        ci_upper.append(mean_val + 1.96 * std_val)
        ci_lower.append(mean_val - 1.96 * std_val)
    
    # 개별 에피소드 궤적을 낮은 투명도로 표시
    for values in all_portfolio_values:
        plt.plot(range(len(values)), values, alpha=0.15, color='gray', linestyle='--',
                 label='Individual Episodes' if values is all_portfolio_values[0] else "")
    
    # 평균 및 신뢰구간 그래프 표시
    x_positions = np.arange(max_days)
    plt.plot(x_positions, mean_values, 'b-', linewidth=2, label='Mean Portfolio Value')
    plt.fill_between(x_positions, ci_lower, ci_upper, color='b', alpha=0.2, label='95% CI')
    
    plt.title('Bootstrap Analysis of Portfolio Performance')
    plt.xlabel('Trading Day')
    plt.ylabel('Portfolio Value ($)')
    
    # x축 눈금을 실제 거래일로 설정
    tick_positions = np.arange(0, max_days, max(1, max_days // 5))
    plt.xticks(tick_positions, [f"Day {i+1}" for i in tick_positions], rotation=45)
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("\nValidation Results Summary:")
    print("==========================")
    print(f"Initial Portfolio Value: ${config['initial_capital']:.2f}")
    print(f"Average Final Portfolio Value: ${mean_values[-1]:.2f}")
    print(f"Standard Deviation: ${np.std([values[-1] for values in all_portfolio_values]):.2f}")
    print(f"95% Confidence Interval: ${ci_lower[-1]:.2f} to ${ci_upper[-1]:.2f}")
    max_return = max([values[-1] for values in all_portfolio_values]) - config["initial_capital"]
    min_return = min([values[-1] for values in all_portfolio_values]) - config["initial_capital"]
    print(f"Best Return: ${max_return:.2f} ({(max_return/config['initial_capital'])*100:.1f}%)")
    print(f"Worst Return: ${min_return:.2f} ({(min_return/config['initial_capital'])*100:.1f}%)")
    
    print("\nPrediction completed.")

def validation_module(config):
    """
    Validation function using TrainModule
    TrainModule을 사용한 검증 함수
    """
    # Initialize modules
    data_module = DataModule(
        config["tickers"],
        period=config["data_period"],
        interval=config["data_interval"],
        train_split=config["train_split"]
    )
    train_data, val_data = data_module.get_data()
    
    # Setup validation environment
    val_env = ImprovedStockTradingEnv(
        val_data, config["tickers"],
        initial_capital=config["initial_capital"],
        transaction_cost=config["transaction_cost"],
        window=config["window"],
        max_trades_per_day=config["max_trades_per_day"],
        cash_fraction=config["cash_fraction"]
    )
    
    # Initialize agent
    state_size = len(val_env._get_state())
    action_size = val_env.action_size
    agent = DQNAgent(
        state_size, action_size, device,
        randomness_scale=config["val_randomness"]
    )
    
    # Load trained weights
    agent.model.load_state_dict(torch.load(config["weights_path"], map_location=device, weights_only=False))
    agent.target_model.load_state_dict(agent.model.state_dict())
    
    # Initialize training module for validation
    train_module = TrainModule(
        train_env=None,
        val_env=val_env,
        agent=agent,
        device=device,
        weights_path=config["weights_path"],
        train_log_csv=config["train_log_csv"],
        val_log_csv=config["val_log_csv"],
        max_replay_calls_per_episode=config["max_replay_calls_per_episode"],
        replay_call_interval=config["replay_call_interval"]
    )
    
    # Run validation
    train_module.validate(val_episodes=config["val_episodes"])

def predict_future(config):
    """
    Predict trading actions for the next 10 days
    향후 10일간의 거래 행동을 예측하는 함수
    """
    data_module = DataModule(
        config["tickers"],
        period=config["data_period"],
        interval=config["data_interval"],
        train_split=config["train_split"]
    )
    _, val_data = data_module.get_data()
    
    print("\nFuture Trading Predictions")
    print("=========================")
    print("Predicting trading actions for the next 10 days based on current market state...")
    
    val_env = ImprovedStockTradingEnv(
        val_data, config["tickers"],
        initial_capital=config["initial_capital"],
        transaction_cost=config["transaction_cost"],
        window=config["window"],
        max_trades_per_day=config["max_trades_per_day"],
        cash_fraction=config["cash_fraction"]
    )
    
    state_size = len(val_env._get_state())
    action_size = val_env.action_size
    agent = DQNAgent(
        state_size, action_size, device,
        randomness_scale=config["val_randomness"]
    )
    
    agent.model.load_state_dict(torch.load(config["weights_path"], map_location=device, weights_only=False))
    agent.target_model.load_state_dict(agent.model.state_dict())
    
    state = val_env.reset()
    portfolio_value = config["initial_capital"]
    current_date = pd.Timestamp.now()
    
    for day in range(10):
        future_date = current_date + pd.Timedelta(days=day)
        print(f"\nDay {day+1} ({future_date.strftime('%Y-%m-%d')})")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print("-" * 50)
        
        action = agent.act(state)
        next_state, reward, done, info = val_env.step(action)
        
        buys = []
        sells = []
        for trade in val_env.trades_today:
            if trade['trade_type'] == "buy":
                buys.append(f"BUY  {trade['ticker']:<6} ${trade['dollar_amt']:.2f}")
            elif trade['trade_type'] == "sell":
                sells.append(f"SELL {trade['ticker']:<6} ${trade['dollar_amt']:.2f}")
        
        if not (sells or buys):
            print("No trades recommended for this day")
        else:
            if sells:
                print("\nSell Orders:")
                for i, sell in enumerate(sells, 1):
                    print(f"{i}. {sell}")
            if buys:
                print("\nBuy Orders:")
                for i, buy in enumerate(buys, 1):
                    print(f"{i}. {buy}")
        
        state = next_state
        portfolio_value = info['portfolio_value']
        if done:
            break
    
    print("\nPrediction completed.")

def main():
    """
    Main function with command line interface
    명령줄 인터페이스가 포함된 메인 함수
    """
    print("\nRL Trading System")
    print("1. Train Model")
    print("2. Validate Model")
    print("3. Predict Next 10 Days")
    print("4. Exit")
    choice = input("Select an option (1-4): ")
    
    if choice == "1":
        train(CONFIG)
    elif choice == "2":
        validate(CONFIG)
    elif choice == "3":
        predict_future(CONFIG)
    elif choice == "4":
        return
    else:
        print("Invalid option. Please try again.")
        main()

if __name__ == "__main__":
    #main()
    #train(CONFIG)
    validation_module(CONFIG)
    #predict_future(CONFIG)

# %%
