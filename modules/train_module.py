import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

class TrainModule:
    """
    Training Module for the Trading Agent
    거래 에이전트를 위한 학습 모듈
    
    This module handles:
    - Training process management
    - Model checkpointing
    - Performance visualization
    - Validation process
    - Trading action prediction
    
    이 모듈의 기능:
    - 학습 프로세스 관리
    - 모델 체크포인트 저장
    - 성능 시각화
    - 검증 프로세스
    - 거래 행동 예측
    """
    
    def __init__(self, train_env, val_env, agent, device,
                 weights_path="dqn_weights.pth",
                 train_log_csv="training_trades.csv",
                 val_log_csv="validation_trades.csv",
                 max_replay_calls_per_episode=20,
                 replay_call_interval=10):
        """
        Initialize the Training Module
        학습 모듈 초기화
        
        Args:
            train_env: Training environment
                    학습 환경
            val_env: Validation environment
                   검증 환경
            agent: DQN agent
                 DQN 에이전트
            device: Device to run on (CPU/GPU)
                  실행할 디바이스 (CPU/GPU)
            weights_path (str): Path to save model weights
                            모델 가중치 저장 경로
            train_log_csv (str): Path to save training logs
                              학습 로그 저장 경로
            val_log_csv (str): Path to save validation logs
                            검증 로그 저장 경로
            max_replay_calls_per_episode (int): Maximum number of replay calls per episode
                                            에피소드당 최대 리플레이 호출 횟수
            replay_call_interval (int): Interval between replay calls
                                    리플레이 호출 간격
        """
        self.train_env = train_env
        self.val_env = val_env
        self.agent = agent
        self.device = device
        self.weights_path = weights_path
        self.train_log_csv = train_log_csv
        self.val_log_csv = val_log_csv
        self.max_replay_calls_per_episode = max_replay_calls_per_episode
        self.replay_call_interval = replay_call_interval
        
        # Load pre-trained weights if available
        # 사전 학습된 가중치가 있으면 로드
        if os.path.exists(weights_path):
            self.agent.model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
            self.agent.target_model.load_state_dict(self.agent.model.state_dict())
            print("Loaded pre-trained weights.")
            
    def train(self, episodes=10):
        """
        Train the agent
        에이전트 학습
        
        Args:
            episodes (int): Number of training episodes
                        학습 에피소드 수
        """
        episode_values = []
        training_trade_logs = []
        
        # Clear existing training log file
        pd.DataFrame(columns=['day', 'trade_type', 'ticker', 'dollar_amt', 'trade_price', 'bonus', 'episode_summary', 'episode']).to_csv(self.train_log_csv, index=False)
        
        # Setup interactive plotting
        # 대화형 플롯 설정
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(10,6))
        
        for e in range(episodes):
            state = self.train_env.reset()
            done = False
            total_reward = 0
            step_count = 0
            replay_calls = 0
            
            while not done:
                # Get action and execute
                # 행동 선택 및 실행
                action = self.agent.act(state)
                next_state, reward, done, info = self.train_env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                
                # Periodic replay for learning
                # 주기적 리플레이 학습
                step_count += 1
                if step_count % self.replay_call_interval == 0 and replay_calls < self.max_replay_calls_per_episode:
                    self.agent.replay()
                    replay_calls += 1
                    
                state = next_state
                total_reward += reward
            
            # Process episode results
            # 에피소드 결과 처리
            day_values = self.train_env.daily_history
            if len(day_values) != len(self.train_env.error_bars):
                self.train_env.error_bars.append(0.0)
                
            episode_values.append(day_values[-1])
            print(f"Training Episode {e+1}/{episodes} - Final Asset: ${day_values[-1]:.2f}, Total Reward: {total_reward:.2f}, Epsilon: {self.agent.epsilon:.2f}")
            
            # Log trades
            # 거래 기록
            for tx in self.train_env.transaction_log:
                tx['episode'] = e+1
                training_trade_logs.append(tx)
            
            # Save model and logs
            # 모델과 로그 저장
            torch.save(self.agent.model.state_dict(), self.weights_path)
            pd.DataFrame(training_trade_logs).to_csv(self.train_log_csv, index=False)
            
            # Update plot
            # 플롯 업데이트
            ax.clear()
            ax.errorbar(range(1, len(day_values)+1), day_values, yerr=self.train_env.error_bars, fmt='o-', capsize=4)
            ax.set_title("Training: End Day Portfolio Value")
            ax.set_xlabel("Trading Day")
            ax.set_ylabel("Portfolio Value ($)")
            plt.pause(0.05)
        
        plt.ioff()
        plt.show()
        print("Saved trained weights and trading log.")
        
    def validate(self, val_episodes=5):
        """
        Validate the trained agent
        학습된 에이전트 검증
        
        Args:
            val_episodes (int): Number of validation episodes
                            검증 에피소드 수
        """
        print("\nValidation Phase:")
        print("=======================")
        
        # Clear existing validation log file
        pd.DataFrame(columns=['day', 'trade_type', 'ticker', 'dollar_amt', 'trade_price', 'bonus', 'episode_summary', 'episode']).to_csv(self.val_log_csv, index=False, mode='w')
        
        self.agent.randomness_scale = 7  # Increase randomness for validation
        val_episode_values = []
        validation_trade_logs = []
        all_portfolio_values = []
        
        print("\nRunning validation episodes...")
        for e in range(val_episodes):
            state = self.val_env.reset()
            done = False
            total_reward = 0
            portfolio_values = []
            last_recorded_day = self.val_env.current_day
            
            while not done:
                action = self.agent.act(state)
                next_state, reward, done, info = self.val_env.step(action)
                
                # Record portfolio value when day changes
                if self.val_env.current_day != last_recorded_day:
                    portfolio_values.append(info['portfolio_value'])
                    last_recorded_day = self.val_env.current_day
                
                state = next_state
                total_reward += reward
            
            # Record final portfolio value if not recorded
            if not portfolio_values or portfolio_values[-1] != info['portfolio_value']:
                portfolio_values.append(info['portfolio_value'])
            
            # Process episode results
            day_values = self.val_env.daily_history
            if len(day_values) != len(self.val_env.error_bars):
                self.val_env.error_bars.append(0.0)
                
            val_episode_values.append(day_values[-1])
            all_portfolio_values.append(portfolio_values)
            print(f"Episode {e+1}: Final Portfolio Value: ${portfolio_values[-1]:.2f}")
            
            # Log trades
            for tx in self.val_env.transaction_log:
                tx['episode'] = e+1
                validation_trade_logs.append(tx)
            
            # Save validation logs after each episode
            pd.DataFrame(validation_trade_logs).to_csv(self.val_log_csv, index=False, mode='w')
            
            # Plot episode results
            plt.figure(figsize=(8,4))
            plt.errorbar(range(1, len(day_values)+1), day_values, yerr=self.val_env.error_bars, fmt='o-', capsize=4)
            plt.title(f"Validation Episode {e+1} - Portfolio Value")
            plt.xlabel("Trading Day")
            plt.ylabel("Portfolio Value ($)")
            plt.grid(True)
            plt.show()
        
        print("\nGenerating bootstrap analysis...")
        
        # Bootstrap Analysis
        max_days = len(self.val_env.price_data)
        
        # Add initial capital (Day 0) to all portfolio values
        for idx, values in enumerate(all_portfolio_values):
            all_portfolio_values[idx] = [self.val_env.initial_capital] + values
        
        # Adjust max_days to include Day 0
        max_days += 1
        
        # Pad or trim values if needed
        for idx, values in enumerate(all_portfolio_values):
            if len(values) < max_days:
                all_portfolio_values[idx] = values + [values[-1]] * (max_days - len(values))
            elif len(values) > max_days:
                all_portfolio_values[idx] = values[:max_days]
        
        plt.figure(figsize=(12, 6))
        daily_values = [[] for _ in range(max_days)]
        
        for episode_values in all_portfolio_values:
            for day, value in enumerate(episode_values):
                if day < max_days:
                    daily_values[day].append(value)
        
        mean_values = []
        ci_upper = []
        ci_lower = []
        
        for day_values in daily_values:
            if day_values:
                mean_val = np.mean(day_values)
                std_val = np.std(day_values)
            else:
                mean_val = np.nan
                std_val = 0
            mean_values.append(mean_val)
            ci_upper.append(mean_val + 1.96 * std_val)
            ci_lower.append(mean_val - 1.96 * std_val)
        
        for values in all_portfolio_values:
            plt.plot(range(len(values)), values, alpha=0.15, color='gray', linestyle='--',
                     label='Individual Episodes' if values is all_portfolio_values[0] else "")
        
        x_positions = np.arange(max_days)
        plt.plot(x_positions, mean_values, 'b-', linewidth=2, label='Mean Portfolio Value')
        plt.fill_between(x_positions, ci_lower, ci_upper, color='b', alpha=0.2, label='95% CI')
        
        plt.title('Bootstrap Analysis of Portfolio Performance')
        plt.xlabel('Trading Day')
        plt.ylabel('Portfolio Value ($)')
        
        # x축 눈금을 Day 0부터 시작하도록 설정
        tick_positions = np.arange(0, max_days, max(1, max_days // 5))
        plt.xticks(tick_positions, [f"Day {i}" for i in tick_positions], rotation=45)
        
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        initial_capital = self.val_env.initial_capital
        print("\nValidation Results Summary:")
        print("==========================")
        print(f"Initial Portfolio Value: ${initial_capital:.2f}")
        print(f"Average Final Portfolio Value: ${mean_values[-1]:.2f}")
        print(f"Standard Deviation: ${np.std([values[-1] for values in all_portfolio_values]):.2f}")
        print(f"95% Confidence Interval: ${ci_lower[-1]:.2f} to ${ci_upper[-1]:.2f}")
        max_return = max([values[-1] for values in all_portfolio_values]) - initial_capital
        min_return = min([values[-1] for values in all_portfolio_values]) - initial_capital
        print(f"Best Return: ${max_return:.2f} ({(max_return/initial_capital)*100:.1f}%)")
        print(f"Worst Return: ${min_return:.2f} ({(min_return/initial_capital)*100:.1f}%)")
        
        print("\nPrediction completed.")
        print(f"\nSaved validation trade log to {self.val_log_csv}")
