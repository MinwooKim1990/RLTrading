import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network Architecture
    듀얼링 심층 Q-네트워크 구조
    
    This network separates the estimation of state values and action advantages
    상태 가치와 행동 이점을 분리하여 추정하는 네트워크
    
    Architecture:
    - Common layers (FC1, FC2 with dropouts)
    - Value stream (estimates state value)
    - Advantage stream (estimates action advantages)
    
    구조:
    - 공통 계층 (FC1, FC2와 드롭아웃)
    - 가치 스트림 (상태 가치 추정)
    - 이점 스트림 (행동 이점 추정)
    """
    
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        """
        Initialize the Dueling DQN
        듀얼링 DQN 초기화
        
        Args:
            input_dim (int): Input dimension (state size)
                          입력 차원 (상태 크기)
            output_dim (int): Output dimension (action size)
                           출력 차원 (행동 크기)
            dropout_rate (float): Dropout probability
                               드롭아웃 확률
        """
        super(DuelingDQN, self).__init__()
        
        # Common feature layers
        # 공통 특성 계층
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Value stream
        # 가치 스트림
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        # 이점 스트림
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass through the network
        네트워크 순전파
        
        Args:
            x (torch.Tensor): Input state
                           입력 상태
                           
        Returns:
            torch.Tensor: Q-values for each action
                       각 행동에 대한 Q-값
        """
        # Common layers
        # 공통 계층
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Split into value and advantage streams
        # 가치와 이점 스트림으로 분리
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage
        # 가치와 이점 결합
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class DQNAgent:
    """
    DQN Agent with Experience Replay and Target Network
    경험 재생과 타겟 네트워크를 갖춘 DQN 에이전트
    
    Features:
    - Experience replay memory
    - Target network for stable learning
    - Epsilon-greedy exploration with decay
    - Randomness scaling for exploration
    
    특징:
    - 경험 재생 메모리
    - 안정적 학습을 위한 타겟 네트워크
    - 입실론-탐욕 탐색과 감쇠
    - 탐색을 위한 무작위성 스케일링
    """
    
    def __init__(self, state_size, action_size, device, lr=0.0005, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, batch_size=32, target_update=10, randomness_scale=1):
        """
        Initialize the DQN Agent
        DQN 에이전트 초기화
        
        Args:
            state_size (int): Dimension of state space
                           상태 공간의 차원
            action_size (int): Number of possible actions
                            가능한 행동의 수
            device (torch.device): Device to run the model on (CPU/GPU)
                                모델을 실행할 디바이스 (CPU/GPU)
            lr (float): Learning rate
                     학습률
            gamma (float): Discount factor
                        할인 계수
            epsilon (float): Initial exploration rate
                         초기 탐색률
            epsilon_min (float): Minimum exploration rate
                             최소 탐색률
            epsilon_decay (float): Decay rate for exploration
                               탐색률 감쇠율
            batch_size (int): Size of training batch
                           학습 배치 크기
            target_update (int): Frequency of target network update
                              타겟 네트워크 업데이트 주기
            randomness_scale (float): Scale factor for exploration randomness
                                  탐색 무작위성 스케일 계수
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
        
        # Initialize networks
        # 네트워크 초기화
        self.model = DuelingDQN(state_size, action_size).to(device)
        self.target_model = DuelingDQN(state_size, action_size).to(device)
        self.update_target_network()
        
        # Initialize optimizer
        # 옵티마이저 초기화
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction='none')
    
    def update_target_network(self):
        """
        Update target network by copying weights from online network
        온라인 네트워크의 가중치를 타겟 네트워크로 복사하여 업데이트
        """
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        경험을 재생 메모리에 저장
        
        Args:
            state: Current state
                 현재 상태
            action: Action taken
                  수행한 행동
            reward: Reward received
                  받은 보상
            next_state: Next state
                      다음 상태
            done: Whether episode is done
                에피소드 종료 여부
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Choose action using epsilon-greedy policy with noise
        노이즈가 포함된 입실론-탐욕 정책을 사용하여 행동 선택
        
        Args:
            state: Current state
                 현재 상태
                 
        Returns:
            int: Selected action
                선택된 행동
        """
        # Add noise to state
        # 상태에 노이즈 추가
        noise_std = 0.05 * (self.randomness_scale / 10.0)
        noisy_state = state + np.random.normal(0, noise_std, size=state.shape)
        
        # Calculate effective epsilon
        # 유효 입실론 계산
        effective_epsilon = self.epsilon * (self.randomness_scale / 10.0)
        if np.random.rand() <= effective_epsilon:
            return random.randrange(self.action_size)
        
        # Get Q-values with noise
        # 노이즈가 포함된 Q-값 계산
        state_tensor = torch.FloatTensor(noisy_state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
            q_noise = torch.normal(0, 0.1, size=q_values.shape).to(self.device)
            q_values = q_values + q_noise * (self.randomness_scale / 10.0)
        self.model.train()
        return torch.argmax(q_values[0]).item()
        
    def replay(self):
        """
        Train the network using experience replay
        경험 재생을 사용하여 네트워크 학습
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        # 메모리에서 배치 샘플링
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        # Convert to tensors
        # 텐서로 변환
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        # 현재 Q-값 계산
        q_values = self.model(states).gather(1, actions).squeeze(1)
        
        # Get next Q values using target network
        # 타겟 네트워크를 사용하여 다음 Q-값 계산
        next_actions = torch.argmax(self.model(next_states), dim=1, keepdim=True)
        next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Calculate loss with importance weights
        # 중요도 가중치를 사용하여 손실 계산
        errors = q_values - targets.detach()
        weights = torch.where(rewards < -5, torch.tensor(2.0, device=self.device), torch.tensor(1.0, device=self.device))
        loss = (weights * (errors ** 2)).mean()
        
        # Optimize the model
        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        # 입실론 업데이트
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network if needed
        # 필요한 경우 타겟 네트워크 업데이트
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.update_target_network()
