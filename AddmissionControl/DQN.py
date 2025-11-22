import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Step 1: Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state, dtype=np.float32), 
            np.array(action, dtype=np.int64), 
            np.array(reward, dtype=np.float32), 
            np.array(next_state, dtype=np.float32), 
            np.array(done, dtype=np.bool_)
        )
    
    def __len__(self):
        return len(self.buffer)

# Step 2: Define the DQN Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Step 3: Define the Admission Control Environment
class AdmissionControlEnv:
    def __init__(self):
        # Assumptions (customize based on your answers; using defaults)
        self.max_cpu = 100.0
        self.max_ram = 100.0
        self.max_bw = 100.0
        self.reset()
    
    def reset(self):
        self.current_cpu = self.max_cpu
        self.current_ram = self.max_ram
        self.current_bw = self.max_bw
        self.queue = []  # List of NSRs: each is [revenue, duration, cpu_demand, ram_demand, bw_demand]
        self.active_nsrs = []  # List of [remaining_duration, cpu_usage, ram_usage, bw_usage, revenue]
        self.time_step = 0
        self.total_revenue = 0
        self._generate_nsr()  # Add initial NSR to queue
        return self._get_state()
    
    def _generate_nsr(self):
        # Simulate NSR arrival: revenue 1-10, duration 1-5, demands 1-20 each
        revenue = random.uniform(1, 10)
        duration = random.randint(1, 5)
        cpu_demand = random.uniform(1, 20)
        ram_demand = random.uniform(1, 20)
        bw_demand = random.uniform(1, 20)
        self.queue.append([revenue, duration, cpu_demand, ram_demand, bw_demand])
    
    def _get_state(self):
        # Simplified state: [norm_cpu, norm_ram, norm_bw, queue_len, avg_queue_rev, front_revenue, front_duration, front_cpu, front_ram, front_bw, norm_time]
        if not self.queue:
            front_nsr = [0, 0, 0, 0, 0]  # Dummy if queue empty
        else:
            front_nsr = self.queue[0]
        
        avg_queue_rev = np.mean([nsr[0] for nsr in self.queue]) if self.queue else 0
        state = [
            self.current_cpu / self.max_cpu,
            self.current_ram / self.max_ram,
            self.current_bw / self.max_bw,
            len(self.queue),
            avg_queue_rev,
            front_nsr[0],  # revenue
            front_nsr[1] / 5.0,  # normalized duration
            front_nsr[2] / 20.0,  # normalized demands
            front_nsr[3] / 20.0,
            front_nsr[4] / 20.0,
            self.time_step / 100.0  # Assume max 100 steps per episode
        ]
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        reward = 0
        done = False
        if not self.queue:
            # No NSR, no-op
            return self._get_state(), 0, False
        
        front_nsr = self.queue.pop(0)
        revenue, duration, cpu_demand, ram_demand, bw_demand = front_nsr
        
        if action == 1:  # Accept
            if (self.current_cpu >= cpu_demand and 
                self.current_ram >= ram_demand and 
                self.current_bw >= bw_demand):
                # Accept: allocate and add to active
                self.current_cpu -= cpu_demand
                self.current_ram -= ram_demand
                self.current_bw -= bw_demand
                self.active_nsrs.append([duration, cpu_demand, ram_demand, bw_demand, revenue])
                reward += revenue / duration  # Partial revenue
            else:
                # Infeasible: penalty
                reward -= 10  # Tunable
        
        else:  # Reject
            if (self.current_cpu >= cpu_demand and 
                self.current_ram >= ram_demand and 
                self.current_bw >= bw_demand):
                reward -= 0.1 * revenue  # Small penalty for missing opportunity
        
        # Simulate time passage: update active NSRs
        for nsr in self.active_nsrs[:]:
            nsr[0] -= 1  # Decrease duration
            # Simulate dynamic usage: fluctuate ±10%
            fluctuation = random.uniform(-0.1, 0.1)
            nsr[1] *= (1 + fluctuation)  # cpu
            nsr[2] *= (1 + fluctuation)  # ram
            nsr[3] *= (1 + fluctuation)  # bw
            
            # Add ongoing revenue
            reward += (nsr[4] / (nsr[0] + 1)) * 0.1  # Small trickle
            
            if nsr[0] <= 0:
                # Release resources
                self.current_cpu += nsr[1]
                self.current_ram += nsr[2]
                self.current_bw += nsr[3]
                self.active_nsrs.remove(nsr)
                self.total_revenue += nsr[4]
        
        # Clamp resources
        self.current_cpu = min(self.max_cpu, max(0, self.current_cpu))
        self.current_ram = min(self.max_ram, max(0, self.current_ram))
        self.current_bw = min(self.max_bw, max(0, self.current_bw))
        
        # Penalty for overuse (if fluctuation caused it)
        overuse = max(0, cpu_demand - self.current_cpu) + max(0, ram_demand - self.current_ram) + max(0, bw_demand - self.current_bw)
        reward -= 5 * overuse
        
        # Simulate arrival: 50% chance of new NSR per step
        if random.random() < 0.5:
            self._generate_nsr()
        
        self.time_step += 1
        if self.time_step >= 100:  # Episode length
            done = True
        
        return self._get_state(), reward, done

# Step 4: DQN Agent Training
def train_dqn(episodes=1000, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update=100, lr=0.001):
    env = AdmissionControlEnv()
    state_dim = len(env._get_state())
    action_dim = 2  # Accept/Reject
    
    eval_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(eval_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(eval_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()
    
    epsilon = epsilon_start
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        
        while True:
            # Epsilon-greedy action
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                with torch.no_grad():
                    q_values = eval_net(torch.tensor(state).unsqueeze(0))
                    action = q_values.argmax().item()
            
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            step += 1
            
            # Train if buffer full
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states = torch.tensor(states)
                actions = torch.tensor(actions).unsqueeze(1)
                rewards = torch.tensor(rewards)
                next_states = torch.tensor(next_states)
                dones = torch.tensor(dones)
                
                # Compute Q targets
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    targets = rewards + gamma * next_q * (~dones)
                
                # Current Q
                current_q = eval_net(states).gather(1, actions).squeeze()
                
                # Loss and update
                loss = nn.MSELoss()(current_q, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if step % target_update == 0:
                target_net.load_state_dict(eval_net.state_dict())
            
            if done:
                break
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.2f}")
    
    # Plot rewards
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Rewards')
    plt.show()

# Run training
if __name__ == "__main__":
    train_dqn()