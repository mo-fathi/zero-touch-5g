import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Config
TOTAL_CPU = 300.0
TOTAL_MEM = 150 * 1024 ** 3  # bytes (50 GiB)
SIM_DURATION = 7200 * 24 # Increased to 2 hours for more DQN learning time
GENERATION_PROB = 0.5  # Increased to 0.5 for more arrivals/queue pressure
BATCH_SIZE = 64
TARGET_UPDATE = 100
GAMMA = 0.99
LR = 0.001
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
BUFFER_SIZE = 10000
STATE_DIM = 13  # Fixed based on state features (5 fixed + 8 from NSR QoS)

device = torch.device("cpu")

# DQN Model
class DQN(nn.Module):
    def __init__(self, state_dim=STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

# Helper Functions
def generate_nsr(nsr_id, sim_time):
    nsr = {
        "id": nsr_id,
        "QoS": {
            "L_max_int": round(random.uniform(1, 10), 2),  # internal latency (intra-slice communication)
            "L_max_ext": round(random.uniform(1, 10), 2),  # external latency (inter-slice or to internet)
            "Phi_min_int": round(random.uniform(50, 200), 2),  # minimum internal throughput
            "Phi_min_ext": round(random.uniform(50, 200), 2),  # minimum external throughput
            "P_max_int": round(random.uniform(0, 0.01), 4),  # maximum internal packet loss
            "P_max_ext": round(random.uniform(0, 0.01), 4)   # maximum external packet loss
        },
        "T0": random.randint(5, 15) * 60,  # Lifespan in seconds (will adjust below)
        "revenue": 0,  # Placeholder (will set below)
        "arrival_time": sim_time 
    }
    # Make revenue inversely tied to demand (throughput)
    total_throughput = nsr["QoS"]["Phi_min_int"] + nsr["QoS"]["Phi_min_ext"]
    nsr["revenue"] = max(10, min(100, int(150 - total_throughput / 3)))
    # Make low-revenue NSRs last longer (occupy resources more)
    if nsr["revenue"] < 50:
        nsr["T0"] = int(nsr["T0"] * 1.5)
    return nsr

def get_resources(active_slices):
    used_cpu = sum(data.get('demand_cpu', 0) for data in active_slices.values())
    used_mem = sum(data.get('demand_mem', 0) for data in active_slices.values())
    free_cpu = max(0, TOTAL_CPU - used_cpu)
    free_mem = max(0, TOTAL_MEM - used_mem)
    return free_cpu, free_mem

def get_state(nsr, q_size, sum_revenue, active_slices):
    free_cpu, free_mem = get_resources(active_slices)
    avg_rev = sum_revenue / q_size if q_size > 0 else 0.0
    
    if nsr is None:
        front = [0] * 8
    else:
        q = nsr["QoS"]
        front = [
            nsr["revenue"] / 100,
            nsr["T0"] / 3600,
            q["L_max_int"] / 10,
            q["Phi_min_int"] / 200,
            q["P_max_int"] / 0.01,
            q["L_max_ext"] / 10,
            q["Phi_min_ext"] / 200,
            q["P_max_ext"] / 0.01
        ]
    
    state = np.array([
        free_cpu / 100,
        free_mem / (50 * 1024 ** 3),
        q_size / 50,
        avg_rev / 100,
        len(active_slices) / 20,
        *front
    ], dtype=np.float32)
    return state

def deploy_nsr(nsr, active_slices, sim_time):
    nsr_id = nsr["id"]
    t0 = nsr.get("T0", 3600)
    qos = nsr["QoS"]
    # Updated demand calculation to incorporate both internal and external parameters
    demand_cpu = (qos["Phi_min_int"] + qos["Phi_min_ext"]) * 0.05 + (qos["L_max_int"] + qos["L_max_ext"]) * 0.1
    demand_mem = (qos["Phi_min_int"] + qos["Phi_min_ext"]) * 100 * 1024 ** 2
    
    free_cpu, free_mem = get_resources(active_slices)
    if free_cpu < demand_cpu or free_mem < demand_mem:
        return False
    
    active_slices[nsr_id] = {
        'end_time': sim_time + t0,
        'revenue': nsr['revenue'],
        'demand_cpu': demand_cpu,
        'demand_mem': demand_mem
    }
    return True

# Simulation Function (supports 'dqn' or 'greedy')
def simulate(mode='dqn'):
    pending_nsrs = deque()
    active_slices = {}
    q_size = 0
    sum_revenue = 0.0
    nsr_id_counter = 0
    sim_time = 0
    cumulative_revenue = 0
    step = 0
    deploying_time = 0  # Tracks remaining deployment time (pauses processing)

    if mode == 'dqn':
        eval_net = DQN().to(device)
        target_net = DQN().to(device)
        target_net.load_state_dict(eval_net.state_dict())
        optimizer = optim.Adam(eval_net.parameters(), lr=LR)
        buffer = deque(maxlen=BUFFER_SIZE)
        epsilon = 1.0
    else:
        buffer = None
        epsilon = None

    total_free_cpu = 0.0
    total_free_mem = 0.0
    total_accepted = 0
    total_rejected = 0

    # Data collection for graphs
    time_steps = []
    free_cpu_over_time = []
    free_mem_over_time = []
    active_slices_over_time = []
    q_size_over_time = []
    cum_revenue_over_time = []

    while sim_time < SIM_DURATION:
        time_steps.append(sim_time)

        if random.random() < GENERATION_PROB:
            nsr = generate_nsr(nsr_id_counter, sim_time)
            pending_nsrs.append(nsr)
            q_size += 1
            sum_revenue += nsr['revenue']
            nsr_id_counter += 1

        if deploying_time > 0:
            deploying_time -= 1
        elif pending_nsrs:
            nsr = pending_nsrs[0]  # Peek
            state = get_state(nsr, q_size, sum_revenue, active_slices)
            state_t = torch.tensor(state).unsqueeze(0).to(device) if mode == 'dqn' else None

            # Action selection
            if mode == 'dqn':
                if random.random() < epsilon:
                    action = random.randint(0, 1)
                else:
                    with torch.no_grad():
                        action = eval_net(state_t).argmax().item()
            else:  # greedy
                action = 1  # Always try to accept

            # Demand estimation (updated to use both int and ext)
            qos = nsr["QoS"]
            demand_cpu = (qos["Phi_min_int"] + qos["Phi_min_ext"]) * 0.05 + (qos["L_max_int"] + qos["L_max_ext"]) * 0.1
            demand_mem = (qos["Phi_min_int"] + qos["Phi_min_ext"]) * 100 * 1024 ** 2

            # Execute action
            reward = 0
            deployed = False
            free_cpu, free_mem = get_resources(active_slices)
            if action == 1 and free_cpu >= demand_cpu and free_mem >= demand_mem:
                deployed = deploy_nsr(nsr, active_slices, sim_time)
                if deployed:
                    reward = nsr["revenue"] / (nsr["T0"] / 60)
                    total_accepted += 1
                    deploying_time = random.randint(5, 15)  # Simulate deployment delay
                else:
                    reward = -50
                    total_rejected += 1
            else:
                reward = -0.1 * nsr["revenue"] if free_cpu > 10 else 0
                total_rejected += 1

            # Remove from queue
            pending_nsrs.popleft()
            q_size -= 1
            sum_revenue -= nsr['revenue']

            # Next state
            next_state = get_state(None, q_size, sum_revenue, active_slices)

            # Store and learn (DQN only)
            if mode == 'dqn':
                buffer.append((state, action, reward, next_state, False))
                if len(buffer) > BATCH_SIZE:
                    batch = random.sample(buffer, BATCH_SIZE)
                    s, a, r, s_, _ = map(np.array, zip(*batch))
                    s = torch.tensor(s, dtype=torch.float32).to(device)
                    a = torch.tensor(a, dtype=torch.long).unsqueeze(1).to(device)
                    r = torch.tensor(r, dtype=torch.float32).to(device)
                    s_ = torch.tensor(s_, dtype=torch.float32).to(device)

                    with torch.no_grad():
                        target = r + GAMMA * target_net(s_).max(1)[0]
                    current = eval_net(s).gather(1, a).squeeze()
                    loss = nn.MSELoss()(current, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                step += 1
                if step % TARGET_UPDATE == 0:
                    target_net.load_state_dict(eval_net.state_dict())
                    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        # Cleanup expired slices
        expired = [sid for sid, data in list(active_slices.items()) if sim_time > data['end_time']]
        for sid in expired:
            revenue = active_slices[sid]['revenue']
            cumulative_revenue += revenue
            del active_slices[sid]

        # Collect metrics
        free_cpu, free_mem = get_resources(active_slices)
        total_free_cpu += free_cpu
        total_free_mem += free_mem
        free_cpu_over_time.append(free_cpu)
        free_mem_over_time.append(free_mem / (1024 ** 3))  # GiB
        active_slices_over_time.append(len(active_slices))
        q_size_over_time.append(q_size)
        cum_revenue_over_time.append(cumulative_revenue)

        sim_time += 1

    avg_free_cpu = total_free_cpu / SIM_DURATION
    avg_free_mem_gib = (total_free_mem / SIM_DURATION) / (1024 ** 3)
    avg_util_cpu = (TOTAL_CPU - avg_free_cpu) / TOTAL_CPU * 100
    avg_util_mem = (TOTAL_MEM - total_free_mem / SIM_DURATION) / TOTAL_MEM * 100

    return {
        'revenue': cumulative_revenue,
        'avg_cpu_util': avg_util_cpu,
        'avg_mem_util': avg_util_mem,
        'accepted': total_accepted,
        'rejected': total_rejected,
        'time_steps': time_steps,
        'free_cpu_over_time': free_cpu_over_time,
        'free_mem_over_time': free_mem_over_time,
        'active_slices_over_time': active_slices_over_time,
        'q_size_over_time': q_size_over_time,
        'cum_revenue_over_time': cum_revenue_over_time
    }

# Run and Compare (use same seed for fair comparison)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
dqn_results = simulate('dqn')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
greedy_results = simulate('greedy')

# Print comparison
print("DQN Results:", {k: v for k, v in dqn_results.items() if k not in ['time_steps', 'free_cpu_over_time', 'free_mem_over_time', 'active_slices_over_time', 'q_size_over_time', 'cum_revenue_over_time']})
print("Greedy Results:", {k: v for k, v in greedy_results.items() if k not in ['time_steps', 'free_cpu_over_time', 'free_mem_over_time', 'active_slices_over_time', 'q_size_over_time', 'cum_revenue_over_time']})

# Generate comparison graphs
plt.figure(figsize=(12, 10))

# Free CPU
plt.subplot(3, 2, 1)
plt.plot(dqn_results['time_steps'], dqn_results['free_cpu_over_time'], label='DQN')
plt.plot(greedy_results['time_steps'], greedy_results['free_cpu_over_time'], label='Greedy')
plt.xlabel('Time (seconds)')
plt.ylabel('Free CPU (cores)')
plt.title('Free CPU Over Time')
plt.legend()
plt.grid(True)

# Free Memory
plt.subplot(3, 2, 2)
plt.plot(dqn_results['time_steps'], dqn_results['free_mem_over_time'], label='DQN')
plt.plot(greedy_results['time_steps'], greedy_results['free_mem_over_time'], label='Greedy')
plt.xlabel('Time (seconds)')
plt.ylabel('Free Memory (GiB)')
plt.title('Free Memory Over Time')
plt.legend()
plt.grid(True)

# Active Slices
plt.subplot(3, 2, 3)
plt.plot(dqn_results['time_steps'], dqn_results['active_slices_over_time'], label='DQN')
plt.plot(greedy_results['time_steps'], greedy_results['active_slices_over_time'], label='Greedy')
plt.xlabel('Time (seconds)')
plt.ylabel('Number of Active Slices')
plt.title('Active Slices Over Time')
plt.legend()
plt.grid(True)

# Queue Size
plt.subplot(3, 2, 4)
plt.plot(dqn_results['time_steps'], dqn_results['q_size_over_time'], label='DQN')
plt.plot(greedy_results['time_steps'], greedy_results['q_size_over_time'], label='Greedy')
plt.xlabel('Time (seconds)')
plt.ylabel('Queue Size')
plt.title('Queue Size Over Time')
plt.legend()
plt.grid(True)

# Cumulative Revenue
plt.subplot(3, 2, 5)
plt.plot(dqn_results['time_steps'], dqn_results['cum_revenue_over_time'], label='DQN')
plt.plot(greedy_results['time_steps'], greedy_results['cum_revenue_over_time'], label='Greedy')
plt.xlabel('Time (seconds)')
plt.ylabel('Cumulative Revenue')
plt.title('Cumulative Revenue Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('comparison_graphs.png')
print("Comparison graphs saved to 'comparison_graphs.png'")