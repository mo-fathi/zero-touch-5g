# agents.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = DQN(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            next_state_t = torch.from_numpy(next_state).float().unsqueeze(0)

            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state_t)).item()

            target_f = self.model(state_t).clone()
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state_t), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class GreedyAgent:
    def act(self, state):
        return 1  # Always accept if possible