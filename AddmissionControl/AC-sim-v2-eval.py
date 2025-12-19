import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
import matplotlib.pyplot as plt

# ==================== AdmissionEnv ==================== 
# (Paste the full AdmissionEnv class here - exactly as in your previous code)
# Make sure it includes the _dequeue_next, generate_nsr, etc.

class AdmissionEnv(gym.Env):
    def __init__(self, queue_length=10, total_resources={'cpu': 1000.0, 'mem': 10000.0, 'bw': 10000.0}, penalty=-10.0, reject_penalty=-0.1, arrival_rate=1.0):
        super(AdmissionEnv, self).__init__()
        self.queue_length = queue_length
        self.total_resources = {k: float(v) for k, v in total_resources.items()}  # Ensure float
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
        revenue = random.uniform(5, 100)
        bw_req = Phi_min_int + Phi_min_ext
        cpu_req = 100 / L_max_int + 100 / L_max_ext
        mem_req = 100 / P_max_int + 100 / P_max_ext
        resources = {'cpu': cpu_req, 'mem': mem_req, 'bw': bw_req}
        QoS = {
            "L_max_int": L_max_int, "L_max_ext": L_max_ext,
            "Phi_min_int": Phi_min_int, "Phi_min_ext": Phi_min_ext,
            "P_max_int": P_max_int, "P_max_ext": P_max_ext
        }
        return {"QoS": QoS, "T0": T0, "revenue": revenue, "resources": resources}

    def get_nsr_features(self, nsr):
        q = nsr['QoS']
        return np.array([q['L_max_int'], q['L_max_ext'], q['Phi_min_int'], q['Phi_min_ext'],
                         q['P_max_int'], q['P_max_ext'], nsr['T0'], nsr['revenue']], dtype=np.float32)

    def get_remaining_res(self):
        return np.array([self.remaining_resources['cpu'], self.remaining_resources['mem'],
                         self.remaining_resources['bw']], dtype=np.float32)

    def get_avg_queue(self):
        if len(self.queue) == 0:
            return 0.0, 0.0, np.zeros(6, dtype=np.float32)
        revs = [nsr['revenue'] for nsr in self.queue]
        T0s = [nsr['T0'] for nsr in self.queue]
        QoSs = np.array([[nsr['QoS'][k] for k in ['L_max_int', 'L_max_ext', 'Phi_min_int',
                                                 'Phi_min_ext', 'P_max_int', 'P_max_ext']]
                         for nsr in self.queue], dtype=np.float32)
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
        revenue_gained = 0.0
        if action == 0:  # Reject
            reward = self.reject_penalty
        else:  # Accept
            req = self.current_nsr['resources']
            enough = all(self.remaining_resources[k] >= req[k] for k in ['cpu', 'mem', 'bw'])
            if enough:
                for k in ['cpu', 'mem', 'bw']:
                    self.remaining_resources[k] -= req[k]
                reward = self.current_nsr['revenue'] / self.current_nsr['T0']
                revenue_gained = self.current_nsr['revenue']
                release_time = self.current_time + self.current_nsr['T0']
                heapq.heappush(self.release_heap, (release_time, self.current_nsr['resources'].copy()))
            else:
                reward = self.penalty
        self.current_nsr = self._dequeue_next()
        if self.current_nsr is None:
            self.done = True
            next_state = np.zeros(19, dtype=np.float32)
        else:
            avg_rev, avg_T0, avg_QoS = self.get_avg_queue()
            nsr_feat = self.get_nsr_features(self.current_nsr)
            res_feat = self.get_remaining_res()
            next_state = np.concatenate((nsr_feat, res_feat, [avg_rev, avg_T0], avg_QoS))
        return next_state, reward, self.done, False, {"revenue_gained": revenue_gained}

# ==================== DQN Model ====================
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

# ==================== Agents ====================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)

    def act(self, state, epsilon=0.0):
        if np.random.rand() <= epsilon:
            return random.randrange(2)
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

class GreedyAgent:
    def act(self, state):
        return 1  # Always try to accept

# ==================== Evaluation Function ====================
def evaluate_agent(env, agent, num_episodes=100, is_dqn=True):
    total_rewards = []
    total_revenues = []
    avg_utilizations = []  # List of (cpu%, mem%, bw%) averages per episode

    for e in range(num_episodes):
        seed = 1000 + e  # Different from training seeds for fair testing
        state, _ = env.reset(seed=seed)
        episode_reward = 0.0
        episode_revenue = 0.0
        utilization_history = []  # List of utilization at each step

        done = False
        while not done:
            action = agent.act(state, epsilon=0.0) if is_dqn else agent.act(state)
            next_state, reward, done, _, info = env.step(action)
            episode_reward += reward
            episode_revenue += info.get("revenue_gained", 0.0)

            # Compute current utilization
            used = {
                'cpu': env.total_resources['cpu'] - env.remaining_resources['cpu'],
                'mem': env.total_resources['mem'] - env.remaining_resources['mem'],
                'bw': env.total_resources['bw'] - env.remaining_resources['bw']
            }
            util = {
                'cpu': used['cpu'] / env.total_resources['cpu'] * 100,
                'mem': used['mem'] / env.total_resources['mem'] * 100,
                'bw': used['bw'] / env.total_resources['bw'] * 100
            }
            utilization_history.append((util['cpu'], util['mem'], util['bw']))

            state = next_state

        total_rewards.append(episode_reward)
        total_revenues.append(episode_revenue)
        if utilization_history:
            avg_util = np.mean(utilization_history, axis=0)
            avg_utilizations.append(avg_util)
        else:
            avg_utilizations.append((0, 0, 0))

    return total_rewards, total_revenues, avg_utilizations

# ==================== Main Evaluation ====================
if __name__ == "__main__":
    env = AdmissionEnv(queue_length=20, arrival_rate=0.1)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Load trained DQN agent
    dqn_agent = DQNAgent(state_size, action_size)
    try:
        dqn_agent.model.load_state_dict(torch.load('dqn_agent.pth'))
        dqn_agent.model.eval()
        print("Successfully loaded 'dqn_agent.pth'")
    except FileNotFoundError:
        print("Error: 'dqn_agent.pth' not found. Train the agent first!")
        exit(1)

    greedy_agent = GreedyAgent()

    print("Evaluating agents over 100 test episodes...")
    dqn_rewards, dqn_revenues, dqn_utils = evaluate_agent(env, dqn_agent, num_episodes=100, is_dqn=True)
    greedy_rewards, greedy_revenues, greedy_utils = evaluate_agent(env, greedy_agent, num_episodes=100, is_dqn=False)

    # Convert utilizations to arrays for easier plotting
    dqn_utils = np.array(dqn_utils)  # (100, 3)
    greedy_utils = np.array(greedy_utils)

    # =============== Plotting ===============
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Total Reward
    axs[0, 0].plot(dqn_rewards, label='DQN', alpha=0.8)
    axs[0, 0].plot(greedy_rewards, label='Greedy', alpha=0.8)
    axs[0, 0].set_title('Total Reward per Episode')
    axs[0, 0].set_xlabel('Test Episode')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Cumulative Revenue
    axs[0, 1].plot(dqn_revenues, label='DQN', alpha=0.8)
    axs[0, 1].plot(greedy_revenues, label='Greedy', alpha=0.8)
    axs[0, 1].set_title('Total Revenue Achieved per Episode')
    axs[0, 1].set_xlabel('Test Episode')
    axs[0, 1].set_ylabel('Revenue')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Average Resource Utilization (stacked or separate lines)
    resources = ['CPU', 'Memory', 'Bandwidth']
    for i, res in enumerate(resources):
        axs[1, 0].plot(dqn_utils[:, i], label=f'DQN {res}' if i == 0 else None, linestyle='-', alpha=0.7)
        axs[1, 0].plot(greedy_utils[:, i], label=f'Greedy {res}' if i == 0 else None, linestyle='--', alpha=0.7)
    axs[1, 0].set_title('Average Resource Utilization (%) per Episode')
    axs[1, 0].set_xlabel('Test Episode')
    axs[1, 0].set_ylabel('Utilization (%)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. Box plot summary of utilization
    data = [dqn_utils.mean(axis=0), greedy_utils.mean(axis=0)]
    labels = ['DQN', 'Greedy']
    x = np.arange(len(labels))
    width = 0.25
    for i in range(3):
        offset = i * width - width
        axs[1, 1].bar(x + offset, [data[0][i], data[1][i]], width, label=resources[i])
    axs[1, 1].set_title('Mean Resource Utilization Across All Episodes')
    axs[1, 1].set_ylabel('Utilization (%)')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels)
    axs[1, 1].legend()
    axs[1, 1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('dqn_vs_greedy_evaluation.png', dpi=200)
    # plt.show()

    # Print summary
    print("\n=== Summary ===")
    print(f"DQN    - Avg Reward: {np.mean(dqn_rewards):.2f}, Avg Revenue: {np.mean(dqn_revenues):.2f}")
    print(f"Greedy - Avg Reward: {np.mean(greedy_rewards):.2f}, Avg Revenue: {np.mean(greedy_revenues):.2f}")
    print(f"DQN    - Avg Util: CPU {dqn_utils.mean(axis=0)[0]:.1f}%, Mem {dqn_utils.mean(axis=0)[1]:.1f}%, BW {dqn_utils.mean(axis=0)[2]:.1f}%")
    print(f"Greedy - Avg Util: CPU {greedy_utils.mean(axis=0)[0]:.1f}%, Mem {greedy_utils.mean(axis=0)[1]:.1f}%, BW {greedy_utils.mean(axis=0)[2]:.1f}%")