# %%
import torch
from modules.data_module import get_stock_data, DEFAULT_TICKERS
from modules.env_module import ImprovedStockTradingEnv
from modules.agent_module import DQNAgent
from modules.train_module import train_agent, validate_agent

def main():
    """
    주식 거래 강화학습 메인 실행 함수
    1. 데이터 준비
    2. 학습/검증 환경 설정
    3. 에이전트 학습
    4. 학습된 에이전트 검증
    """
    # CUDA 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 로드
    print("\n데이터 다운로드 중...")
    train_data, val_data = get_stock_data(DEFAULT_TICKERS)
    print(f"학습 데이터 크기: {train_data.shape}")
    print(f"검증 데이터 크기: {val_data.shape}")
    
    # 학습 환경 설정
    print("\n학습 환경 초기화 중...")
    train_env = ImprovedStockTradingEnv(
        price_data=train_data,
        initial_capital=10000,
        transaction_cost=0.01,
        window=5,
        risk_penalty=50,
        trade_penalty=5,
        validation_mode=False
    )
    
    # 상태/행동 공간 크기 계산
    state_size = len(train_env._get_state())
    action_size = train_env.action_size
    print(f"상태 공간 크기: {state_size}")
    print(f"행동 공간 크기: {action_size}")
    
    # DQN 에이전트 초기화
    print("\nDQN 에이전트 초기화 중...")
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        device=device,
        lr=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        target_update=10
    )
    
    # 에이전트 학습
    print("\n학습 시작...")
    episodes = 100
    train_agent(train_env, agent, episodes)
    
    # 검증 환경 설정
    print("\n검증 환경 초기화 중...")
    val_env = ImprovedStockTradingEnv(
        price_data=val_data,
        initial_capital=10000,
        transaction_cost=0.01,
        window=5,
        risk_penalty=50,
        trade_penalty=5,
        validation_mode=True,
        prediction_penalty=50
    )
    
    # 학습된 에이전트 검증
    print("\n검증 시작...")
    validate_agent(val_env, agent)

if __name__ == "__main__":
    main() 
# %%
