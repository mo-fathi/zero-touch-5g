import random
import time
from collections import deque

# Cluster configuration (simulated Kind nodes)
nodes = {
    "V1": {"CPU": 50, "RAM": 100},  # Worker nodes
    "V2": {"CPU": 50, "RAM": 100},
    "V3": {"CPU": 50, "RAM": 100}
}
total_nodes = len(nodes)

# NSR queue (Lambda)
lambda_queue = deque()

# Active slices tracking
active_slices = {}

# Generate a new NSR and add to queue
def generate_nsr():
    nsr_id = random.randint(1, 1000)  # Unique ID to avoid conflicts
    nsr = {
        "id": nsr_id,
        "VNFs": [
            {"CPU": random.uniform(5, 20), "RAM": random.uniform(10, 40)} for _ in range(random.randint(1, 3))
        ],
        "QoS": {"bandwidth": random.uniform(50, 200), "latency": random.uniform(1, 10)},
        "T0": random.uniform(10, 50)  # Lifespan in seconds
    }
    lambda_queue.append(nsr)
    print(f"New NSR_{nsr_id} added to queue at time {current_time:.1f}s, T0={nsr['T0']:.1f}s")

# AC Agent (Greedy heuristic)
def admission_control():
    global current_time
    if not lambda_queue:
        return

    nsr = lambda_queue.popleft()  # FCFS
    total_cpu = sum(vnf["CPU"] for vnf in nsr["VNFs"])
    total_ram = sum(vnf["RAM"] for vnf in nsr["VNFs"])

    # Check resource availability
    can_deploy = False
    for node in nodes:
        if total_cpu <= nodes[node]["CPU"] and total_ram <= nodes[node]["RAM"]:
            can_deploy = True
            nodes[node]["CPU"] -= total_cpu
            nodes[node]["RAM"] -= total_ram
            break

    if can_deploy:
        active_slices[nsr["id"]] = {"time_left": nsr["T0"], "resources": {"CPU": total_cpu, "RAM": total_ram}, "node": node}
        print(f"Accepted NSR_{nsr['id']} at time {current_time:.1f}s on {node}, T0={nsr['T0']:.1f}s")
    else:
        print(f"Rejected NSR_{nsr['id']} at time {current_time:.1f}s due to insufficient resources")

# Continuous simulation loop
current_time = 0
print("Simulation started at 08:49 PM CEST, August 24, 2025...")
while True:
    # Generate new NSR at random interval (1-5 seconds)
    if random.random() < 0.3:  # 30% chance per second to generate NSR
        generate_nsr()

    # Process AC
    admission_control()

    # Update time and manage slice expiration
    current_time += 1
    for slice_id, details in list(active_slices.items()):
        details["time_left"] -= 1
        if details["time_left"] <= 0:
            node = details["node"]
            nodes[node]["CPU"] += details["resources"]["CPU"]
            nodes[node]["RAM"] += details["resources"]["RAM"]
            del active_slices[slice_id]
            print(f"Slice NSR_{slice_id} expired at time {current_time:.1f}s on {node}")

    # Print current state every 5 seconds
    if current_time % 5 == 0:
        print(f"\nTime: {current_time:.1f}s | Queue size: {len(lambda_queue)} | Active slices: {len(active_slices)}")
        print(f"Remaining resources: {nodes}")
        print("-" * 50)

    # Slow down simulation for readability (1 second real-time per step)
    time.sleep(1)