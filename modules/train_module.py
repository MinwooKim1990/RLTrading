import torch
import matplotlib.pyplot as plt
import numpy as np

def train_agent(train_env, agent, episodes, plot=True):
    """
    에이전트 학습 함수
    
    Args:
        train_env: 학습 환경
        agent: DQN 에이전트
        episodes: 학습 에피소드 수
        plot: 실시간 시각화 여부
    
    Returns:
        list: 각 에피소드별 최종 포트폴리오 가치
    """
    episode_values = []
    
    if plot:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))
    
    for e in range(episodes):
        state = train_env.reset()
        done = False
        daily_values = [train_env.portfolio_history[0]]
        total_reward = 0
        
        while not done:
            # 행동 선택 및 환경과 상호작용
            action = agent.act(state)
            next_state, reward, done, info = train_env.step(action)
            
            # 경험 저장 및 학습
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            daily_values.append(info['portfolio_value'])
        
        agent.replay()
        episode_values.append(daily_values[-1])
        
        # 학습 진행 상황 출력
        print(f"Training Episode {e+1}/{episodes} - Final Asset: ${daily_values[-1]:.2f}, "
              f"Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        if plot:
            # 일별 포트폴리오 가치 그래프
            ax1.plot(daily_values, label=f"Episode {e+1}")
            ax1.set_title("Training: Daily Portfolio Value")
            ax1.set_xlabel("Trading Day")
            ax1.set_ylabel("Portfolio Value ($)")
            ax1.legend(fontsize='small', loc='upper left')
            
            # 에피소드별 최종 포트폴리오 가치 그래프
            ax2.clear()
            ax2.plot(episode_values, marker='o')
            ax2.set_title("Training: Final Portfolio Value per Episode")
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Final Portfolio Value ($)")
            plt.pause(0.05)
    
    if plot:
        plt.ioff()
        plt.show()
    
    return episode_values

def validate_agent(val_env, agent, episodes=5):
    """
    학습된 에이전트 검증
    
    Args:
        val_env: 검증 환경
        agent: 학습된 DQN 에이전트
        episodes: 검증 에피소드 수
    
    Returns:
        list: 각 에피소드별 최종 포트폴리오 가치
    """
    val_episode_values = []
    
    for e in range(episodes):
        state = val_env.reset()
        done = False
        daily_values = [val_env.portfolio_history[0]]
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = val_env.step(action)
            state = next_state
            total_reward += reward
            daily_values.append(info['portfolio_value'])
            
        val_episode_values.append(daily_values[-1])
        
        print(f"Validation Episode {e+1} - Final Asset: ${daily_values[-1]:.2f}, "
              f"Total Reward: {total_reward:.2f}")
        
        # 검증 결과 시각화
        plt.figure(figsize=(8,4))
        plt.plot(daily_values, marker='o')
        plt.title(f"Validation Episode {e+1} - Daily Portfolio Value")
        plt.xlabel("Trading Day")
        plt.ylabel("Portfolio Value ($)")
        plt.show()
    
    return val_episode_values

def plot_training_results(episode_values, daily_values=None):
    """
    학습 결과 시각화
    
    Args:
        episode_values: 에피소드별 최종 포트폴리오 가치
        daily_values: 일별 포트폴리오 가치 (선택사항)
    """
    plt.figure(figsize=(12,6))
    
    if daily_values is not None:
        plt.subplot(1,2,1)
        plt.plot(daily_values)
        plt.title("Daily Portfolio Value")
        plt.xlabel("Trading Day")
        plt.ylabel("Portfolio Value ($)")
        
        plt.subplot(1,2,2)
    
    plt.plot(episode_values, marker='o')
    plt.title("Final Portfolio Value per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Final Portfolio Value ($)")
    plt.show() 