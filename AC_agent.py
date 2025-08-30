import random
import time
from kafka import KafkaConsumer
import json
import numpy as np

# Cluster configuration (simulated Kind nodes)
nodes = {
    "V1": {"type": "core", "CPU": 50, "RAM": 100},
    "V2": {"type": "core", "CPU": 50, "RAM": 100},
    "V3": {"type": "ran", "CPU": 50, "RAM": 100, "BW": 100}
}

# Active slices tracking
active_slices = {}

# Q-learning parameters
q_table = {}  # State-action pairs
alpha = 0.1   # Learning rate
gamma = 0.9   # Discount factor
epsilon = 0.1 # Exploration rate

# Kafka configuration
bootstrap_servers = ['localhost:9092']
topic = 'lambda'

# Initialize Kafka consumer
consumer = KafkaConsumer(
    topic,
    bootstrap_servers=bootstrap_servers,
    auto_offset_reset='latest',
    enable_auto_commit=True,
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

# Get state: (CPU_core, RAM_core, CPU_ran, RAM_ran, BW_ran, queue_length)
def get_state():
    cpu_core = sum(nodes[n]["CPU"] for n in nodes if nodes[n]["type"] == "core")
    ram_core = sum(nodes[n]["RAM"] for n in nodes if nodes[n]["type"] == "core")
    cpu_ran = sum(nodes[n]["CPU"] for n in nodes if nodes[n]["type"] == "ran")
    ram_ran = sum(nodes[n]["RAM"] for n in nodes if nodes[n]["type"] == "ran")
    bw_ran = sum(nodes[n]["BW"] for n in nodes if nodes[n]["type"] == "ran")
    queue_len = len([msg for msg in consumer])  # Approximate queue length
    return (cpu_core, ram_core, cpu_ran, ram_ran, bw_ran, queue_len)

# Q-learning action selection
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 1)  # Explore
    else:
        if state not in q_table:
            q_table[state] = [0, 0]  # Initialize for accept (1) and reject (0)
        return np.argmax(q_table[state])  # Exploit

# AC Agent with Q-learning
def admission_control(nsr):
    global current_time
    state = get_state()
    action = choose_action(state)

    if action == 1:  # Attempt to accept
        total_cpu_core = sum(vnf["CPU"] for vnf in nsr["VNFs"] if vnf["is_core"])
        total_ram_core = sum(vnf["RAM"] for vnf in nsr["VNFs"] if vnf["is_core"])
        total_cpu_ran = sum(vnf["CPU"] for vnf in nsr["VNFs"] if not vnf["is_core"])
        total_ram_ran = sum(vnf["RAM"] for vnf in nsr["VNFs"] if not vnf["is_core"])
        bw_required = nsr["BW"]

        can_deploy = False
        core_node = next((n for n in nodes if nodes[n]["type"] == "core" and
                         total_cpu_core <= nodes[n]["CPU"] and total_ram_core <= nodes[n]["RAM"]), None)
        ran_node = next((n for n in nodes if nodes[n]["type"] == "ran" and
                        total_cpu_ran <= nodes[n]["CPU"] and total_ram_ran <= nodes[n]["RAM"] and
                        bw_required <= nodes[n]["BW"]), None)

        if core_node and ran_node:
            can_deploy = True
            nodes[core_node]["CPU"] -= total_cpu_core
            nodes[core_node]["RAM"] -= total_ram_core
            nodes[ran_node]["CPU"] -= total_cpu_ran
            nodes[ran_node]["RAM"] -= total_ram_ran
            nodes[ran_node]["BW"] -= bw_required

        if can_deploy:
            active_slices[nsr["id"]] = {
                "time_left": nsr["T0"],
                "resources": {"CPU_core": total_cpu_core, "RAM_core": total_ram_core,
                             "CPU_ran": total_cpu_ran, "RAM_ran": total_ram_ran, "BW": bw_required},
                "nodes": {"core": core_node, "ran": ran_node}
            }
            reward = 10  # Positive reward for successful acceptance
            print(f"Accepted NSR_{nsr['id']} at time {current_time:.1f}s on {core_node} (core) and {ran_node} (ran), T0={nsr['T0']:.1f}s")
        else:
            reward = -5  # Penalty for attempting infeasible acceptance
            print(f"Rejected NSR_{nsr['id']} at time {current_time:.1f}s due to insufficient resources")
    else:
        reward = -5  # Penalty for rejection
        print(f"Rejected NSR_{nsr['id']} at time {current_time:.1f}s by Q-learning")

    # Update Q-value
    next_state = get_state()
    if state not in q_table:
        q_table[state] = [0, 0]
    if next_state not in q_table:
        q_table[next_state] = [0, 0]
    old_value = q_table[state][action]
    next_max = max(q_table[next_state])
    q_table[state][action] += alpha * (reward + gamma * next_max - old_value)

# Continuous simulation loop
current_time = 0
print("AC Agent started at 11:13 PM CEST, August 30, 2025...")
for message in consumer:
    nsr = message.value
    admission_control(nsr)

    # Update time and manage slice expiration
    current_time += 1
    for slice_id, details in list(active_slices.items()):
        details["time_left"] -= 1
        if details["time_left"] <= 0:
            nodes[details["nodes"]["core"]]["CPU"] += details["resources"]["CPU_core"]
            nodes[details["nodes"]["core"]]["RAM"] += details["resources"]["RAM_core"]
            nodes[details["nodes"]["ran"]]["CPU"] += details["resources"]["CPU_ran"]
            nodes[details["nodes"]["ran"]]["RAM"] += details["resources"]["RAM_ran"]
            nodes[details["nodes"]["ran"]]["BW"] += details["resources"]["BW"]
            del active_slices[slice_id]
            print(f"Slice NSR_{slice_id} expired at time {current_time:.1f}s")

    # Print current state every 5 seconds
    if current_time % 5 == 0:
        state = get_state()
        print(f"\nTime: {current_time:.1f}s | Active slices: {len(active_slices)}")
        print(f"Remaining resources: {nodes}")
        print(f"State: {state}")
        print(f"Q-table size: {len(q_table)}")
        print("-" * 50)

    time.sleep(1)  # Simulate 1 second per step