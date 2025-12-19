import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import heapq

class AdmissionEnv(gym.Env):
    def __init__(self, queue_length=10, total_resources={'cpu': 1000.0, 'mem': 10000.0, 'bw': 10000.0}, penalty=-10.0, reject_penalty=-0.1, arrival_rate=1.0):
        super(AdmissionEnv, self).__init__()
        self.queue_length = queue_length
        self.total_resources = total_resources
        self.penalty = penalty
        self.reject_penalty = reject_penalty
        self.arrival_rate = arrival_rate
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(19,), dtype=np.float32)

    def generate_nsr(self):
        L_max_int = random.uniform(1, 100)
        L_max_ext = random.uniform(1, 100)
        Phi_min_int = random.uniform(1, 50)
        Phi_min_ext = random.uniform(1, 50)
        P_max_int = random.uniform(0.001, 0.05)
        P_max_ext = random.uniform(0.001, 0.05)
        T0 = random.uniform(10, 500)
        revenue = random.uniform(50, 1000)
        # Compute resource requirements based on QoS (arbitrary but reasonable mappings)
        bw_req = Phi_min_int + Phi_min_ext
        cpu_req = 100 / L_max_int + 100 / L_max_ext  # More CPU for stricter (lower) latency
        mem_req = 100 / P_max_int + 100 / P_max_ext  # More memory for stricter (lower) packet loss
        resources = {'cpu': cpu_req, 'mem': mem_req, 'bw': bw_req}
        QoS = {
            "L_max_int": L_max_int,
            "L_max_ext": L_max_ext,
            "Phi_min_int": Phi_min_int,
            "Phi_min_ext": Phi_min_ext,
            "P_max_int": P_max_int,
            "P_max_ext": P_max_ext
        }
        return {"QoS": QoS, "T0": T0, "revenue": revenue, "resources": resources}

    def get_nsr_features(self, nsr):
        q = nsr['QoS']
        return np.array([q['L_max_int'], q['L_max_ext'], q['Phi_min_int'], q['Phi_min_ext'], q['P_max_int'], q['P_max_ext'], nsr['T0'], nsr['revenue']], dtype=np.float32)

    def get_remaining_res(self):
        return np.array([self.remaining_resources['cpu'], self.remaining_resources['mem'], self.remaining_resources['bw']], dtype=np.float32)

    def get_avg_queue(self):
        if len(self.queue) == 0:
            return 0.0, 0.0, np.zeros(6, dtype=np.float32)
        revs = [nsr['revenue'] for nsr in self.queue]
        T0s = [nsr['T0'] for nsr in self.queue]
        QoSs = np.array([[nsr['QoS'][k] for k in ['L_max_int', 'L_max_ext', 'Phi_min_int', 'Phi_min_ext', 'P_max_int', 'P_max_ext']] for nsr in self.queue], dtype=np.float32)
        return np.mean(revs), np.mean(T0s), np.mean(QoSs, axis=0)

    def _dequeue_next(self):
        if not self.queue:
            return None
        next_arrival = self.queue[0]['arrival_time']
        while self.release_heap and self.release_heap[0][0] <= next_arrival:
            _, res = heapq.heappop(self.release_heap)
            for k in ['cpu', 'mem', 'bw']:
                self.remaining_resources[k] += res[k]
        self.current_time = next_arrival
        return self.queue.pop(0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.queue = []
        current_arrival = 0.0
        for _ in range(self.queue_length):
            inter = random.expovariate(self.arrival_rate)
            current_arrival += inter
            nsr = self.generate_nsr()
            nsr['arrival_time'] = current_arrival
            self.queue.append(nsr)
        self.remaining_resources = self.total_resources.copy()
        self.current_time = 0.0
        self.release_heap = []
        self.done = False
        self.current_nsr = self._dequeue_next()
        if self.current_nsr is None:
            self.done = True
            return np.zeros(19, dtype=np.float32), {}
        avg_rev, avg_T0, avg_QoS = self.get_avg_queue()
        nsr_feat = self.get_nsr_features(self.current_nsr)
        res_feat = self.get_remaining_res()
        state = np.concatenate((nsr_feat, res_feat, [avg_rev, avg_T0], avg_QoS))
        return state, {}

    def step(self, action):
        reward = 0.0
        if action == 0:  # Reject
            reward = self.reject_penalty
        else:  # Accept
            req = self.current_nsr['resources']
            enough = all(self.remaining_resources[k] >= req[k] for k in ['cpu', 'mem', 'bw'])
            if enough:
                for k in ['cpu', 'mem', 'bw']:
                    self.remaining_resources[k] -= req[k]
                reward = self.current_nsr['revenue'] / self.current_nsr['T0']
                release_time = self.current_time + self.current_nsr['T0']
                heapq.heappush(self.release_heap, (release_time, self.current_nsr['resources'].copy()))
            else:
                reward = self.penalty
        # Dequeue next NSR
        self.current_nsr = self._dequeue_next()
        if self.current_nsr is None:
            self.done = True
            next_state = np.zeros(19, dtype=np.float32)
        else:
            avg_rev, avg_T0, avg_QoS = self.get_avg_queue()
            nsr_feat = self.get_nsr_features(self.current_nsr)
            res_feat = self.get_remaining_res()
            next_state = np.concatenate((nsr_feat, res_feat, [avg_rev, avg_T0], avg_QoS))
        return next_state, reward, self.done, False, {}

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.from_numpy(state).float().unsqueeze(0)
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.from_numpy(state).float().unsqueeze(0)
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            output = self.model(state)
            with torch.no_grad():
                next_q = self.model(next_state)
            target_value = reward if done else (reward + self.gamma * torch.max(next_q).item())
            target = output.clone()
            target[0][action] = target_value
            self.optimizer.zero_grad()
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    env = AdmissionEnv(queue_length=20, arrival_rate=0.1)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")