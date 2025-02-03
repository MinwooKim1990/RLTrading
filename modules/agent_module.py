import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DuelingDQN(nn.Module):
    """
    Dueling DQN 신경망 구조
    - 가치 스트림과 이점 스트림을 분리하여 학습
    """
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 가치 스트림 (상태의 가치를 추정)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 이점 스트림 (각 행동의 이점을 추정)
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
        # Q값 = 가치 + (이점 - 평균 이점)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class DQNAgent:
    """
    DQN 에이전트
    - Double DQN 사용
    - Dueling Network 구조
    - Experience Replay 메모리
    - Epsilon-greedy 탐험
    """
    def __init__(self, state_size, action_size, device, lr=0.0005, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, batch_size=32, target_update=10):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=10000)
        self.gamma = gamma  # 할인율
        self.epsilon = epsilon  # 탐험률
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.learn_step_counter = 0
        
        # 메인 네트워크와 타겟 네트워크
        self.model = DuelingDQN(state_size, action_size).to(device)
        self.target_model = DuelingDQN(state_size, action_size).to(device)
        self.update_target_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def update_target_network(self):
        """타겟 네트워크 업데이트"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """경험 메모리에 저장"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        행동 선택
        - epsilon 확률로 무작위 행동
        - 1-epsilon 확률로 최적 행동
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
        self.model.train()
        return torch.argmax(q_values[0]).item()
    
    def replay(self):
        """
        경험 재생을 통한 학습
        - Double DQN: 행동 선택과 평가를 분리
        - 배치 학습으로 안정성 향상
        """
        if len(self.memory) < self.batch_size:
            return
            
        # 배치 샘플링
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([sample[0] for sample in minibatch]).to(self.device)
        actions = torch.LongTensor([sample[1] for sample in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([sample[2] for sample in minibatch]).to(self.device)
        next_states = torch.FloatTensor([sample[3] for sample in minibatch]).to(self.device)
        dones = torch.FloatTensor([float(sample[4]) for sample in minibatch]).to(self.device)
        
        # 현재 Q값 계산
        q_values = self.model(states).gather(1, actions).squeeze(1)
        
        # Double DQN: 메인 네트워크로 행동 선택, 타겟 네트워크로 평가
        next_actions = torch.argmax(self.model(next_states), dim=1, keepdim=True)
        next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 손실 계산 및 역전파
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 탐험률 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 타겟 네트워크 주기적 업데이트
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.update_target_network() 