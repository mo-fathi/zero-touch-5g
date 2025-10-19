import random
import time
import datetime
from kafka import KafkaConsumer
from kafka import KafkaProducer
import json
import numpy as np
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.utils.quantity import parse_quantity
from decimal import Decimal
import math

# Cluster configuration
# a list of node names 
nodes = []
    

# Active slices tracking
active_slices = {}

# Q-learning parameters
q_table = {}  # State-action pairs
alpha = 0.1   # Learning rate
gamma = 0.9   # Discount factor
epsilon = 1.0 # Exploration rate

# Kafka configuration
bootstrap_servers = ['localhost:9092']
consumer_topic = 'lambda'
producer_topic = 'deploy'

# Initialize Kafka consumer for lambda
consumer = KafkaConsumer(
    consumer_topic,
    bootstrap_servers=bootstrap_servers,
    auto_offset_reset='latest',
    enable_auto_commit=True,
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

# Initialize Kafka producer for lifecycle
producer = KafkaProducer(
    bootstrap_servers=bootstrap_servers,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Convert values of resources
def parse_quantity(quantity_str):
    """
    Convert Kubernetes quantity strings (e.g. '100m', '2Gi') to float values.
    CPU: returns cores as float
    Memory: returns bytes as float
    """
    quantity_str = str(quantity_str).strip()
    if quantity_str.endswith('m'):  # CPU in millicores
        return float(quantity_str[:-1]) / 1000
    elif quantity_str.endswith('n'): # CPU in nanosecond
        return float(quantity_str[:-1]) / 1_000_000_000.0
    elif quantity_str.endswith('Ki'):
        return float(quantity_str[:-2]) * 1024
    elif quantity_str.endswith('Mi'):
        return float(quantity_str[:-2]) * 1024 ** 2
    elif quantity_str.endswith('Gi'):
        return float(quantity_str[:-2]) * 1024 ** 3
    elif quantity_str.endswith('Ti'):
        return float(quantity_str[:-2]) * 1024 ** 4
    else:
        try:
            return float(quantity_str)
        except ValueError:
            return 0.0


def get_realtime_resources():
    # Load configuration
    try:
        config.load_kube_config()        # local dev
    except:
        config.load_incluster_config()   # inside cluster

    v1 = client.CoreV1Api()
    custom_api = client.CustomObjectsApi()

    nodes = v1.list_node().items
    total_alloc_cpu = total_alloc_mem = 0.0
    total_used_cpu = total_used_mem = 0.0

    # Get metrics from Metrics API (requires metrics-server)
    #TODO just worker nodes should be add for this metrics tupple
    metrics = custom_api.list_cluster_custom_object(
        group="metrics.k8s.io",
        version="v1beta1",
        plural="nodes"
    )


    usage_by_node = {
        item["metadata"]["name"]: item["usage"]
        for item in metrics["items"]
    }

    for node in nodes:
        labels = node.metadata.labels or {}
        name = node.metadata.name


        # Skip control-plane / master nodes
        if (
            "node-role.kubernetes.io/control-plane" in labels
            or "node-role.kubernetes.io/master" in labels
        ):
            continue

        alloc = node.status.allocatable
        alloc_cpu = parse_quantity(alloc["cpu"])
        alloc_mem = parse_quantity(alloc["memory"])

        total_alloc_cpu += alloc_cpu
        total_alloc_mem += alloc_mem

        if name in usage_by_node:
            usage = usage_by_node[name]
            used_cpu = parse_quantity(usage["cpu"])
            used_mem = parse_quantity(usage["memory"])
        else:
            used_cpu = used_mem = 0.0

        total_used_cpu += used_cpu
        total_used_mem += used_mem


    free_cpu = total_alloc_cpu - total_used_cpu
    free_mem = total_alloc_mem - total_used_mem

    print("=== Kubernetes Cluster Real-Time Resources ===")
    print(f"Allocatable CPU cores: {total_alloc_cpu:.2f}")
    print(f"Used CPU cores:        {total_used_cpu:.2f}")
    print(f"Free CPU cores:        {free_cpu:.2f}\n")

    print(f"Allocatable Memory:    {total_alloc_mem / (1024 ** 3):.2f} GiB")
    print(f"Used Memory:           {total_used_mem / (1024 ** 3):.2f} GiB")
    print(f"Free Memory:           {free_mem / (1024 ** 3):.2f} GiB")

    return {
        'total_alloc_cpu'   : total_alloc_cpu,
        'total_used_cpu'    : total_used_cpu,
        'free_cpu'          : free_cpu,
        'total_alloc_mem'   : total_alloc_mem,
        'total_used_mem'    : total_used_mem,
        'free_mem'          : free_mem
    }

# Get state: (CPU_core, RAM_core, CPU_ran, RAM_ran, BW_ran, queue_length)
def get_node_state(node_name):
    
# Q-learning action selection
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        # return random.randint(0, 1)  # Explore
        return 1
    else:
        # if state not in q_table:
        #     q_table[state] = [0, 0]  # Initialize for accept (1) and reject (0)
        # return np.argmax(q_table[state])  # Exploit
        return 0
    
def get_resource_metrics():

    # TODO seprate RAN and Core resource metrics
    current_resources = get_realtime_resources()
    cluster_resource_metrics = {
        'remaining_cpu_cores': current_resources['free_cpu'],
        'remaining_ram_cores': current_resources['free_mem'],
        'remaining_cpu_rans': current_resources['free_cpu'],
        'remaining_ram_rans': current_resources['free_mem'],
        'remaining_bandwith_tans': Decimal(0.0)
    }
    
    return cluster_resource_metrics


# AC Agent with Q-learning
def admission_control(nsr):

    # get cluster remaining resource state
    state = get_resource_metrics()
    # add nsr to state
    state['NSR'] = nsr
    # choose action by algorithm 
    action = choose_action(state)

    # print(action)
    
    
    if action == 1:
        # Publish accepted NSR to deploy topic
        print("nsr id ",nsr.get('id', 'default'), "  accepted")
        producer.send(producer_topic, nsr)
        producer.flush()  # Ensure message is sent immediately
        
    else:
        print("nsr id ",nsr.get('id', 'default'), "  rejected")
    # if action == 1:  # Attempt to accept
    #     total_cpu_core = sum(vnf["CPU"] for vnf in nsr["VNFs"] if vnf["is_core"])
    #     total_ram_core = sum(vnf["RAM"] for vnf in nsr["VNFs"] if vnf["is_core"])
    #     total_cpu_ran = sum(vnf["CPU"] for vnf in nsr["VNFs"] if not vnf["is_core"])
    #     total_ram_ran = sum(vnf["RAM"] for vnf in nsr["VNFs"] if not vnf["is_core"])
    #     bw_required = nsr["BW"]

    #     can_deploy = False
    #     core_node = next((n for n in nodes if nodes[n]["type"] == "core" and
    #                      total_cpu_core <= nodes[n]["CPU"] and total_ram_core <= nodes[n]["RAM"]), None)
    #     ran_node = next((n for n in nodes if nodes[n]["type"] == "ran" and
    #                     total_cpu_ran <= nodes[n]["CPU"] and total_ram_ran <= nodes[n]["RAM"] and
    #                     bw_required <= nodes[n]["BW"]), None)

    #     if core_node and ran_node:
    #         can_deploy = True
    #         nodes[core_node]["CPU"] -= total_cpu_core
    #         nodes[core_node]["RAM"] -= total_ram_core
    #         nodes[ran_node]["CPU"] -= total_cpu_ran
    #         nodes[ran_node]["RAM"] -= total_ram_ran
    #         nodes[ran_node]["BW"] -= bw_required

    #     if can_deploy:
    #         active_slices[nsr["id"]] = {
    #             "time_left": nsr["T0"],
    #             "resources": {"CPU_core": total_cpu_core, "RAM_core": total_ram_core,
    #                          "CPU_ran": total_cpu_ran, "RAM_ran": total_ram_ran, "BW": bw_required},
    #             "nodes": {"core": core_node, "ran": ran_node}
    #         }
    #         reward = 10  # Positive reward for successful acceptance
    #         print(f"Accepted NSR_{nsr['id']} at time {current_time:.1f}s on {core_node} (core) and {ran_node} (ran), T0={nsr['T0']:.1f}s")
    #     else:
    #         reward = -5  # Penalty for attempting infeasible acceptance
    #         print(f"Rejected NSR_{nsr['id']} at time {current_time:.1f}s due to insufficient resources")
    # else:
    #     reward = -5  # Penalty for rejection
    #     print(f"Rejected NSR_{nsr['id']} at time {current_time:.1f}s by Q-learning")

    # Update Q-value
    # next_state = get_state()
    # if state not in q_table:
    #     q_table[state] = [0, 0]
    # if next_state not in q_table:
    #     q_table[next_state] = [0, 0]
    # old_value = q_table[state][action]
    # next_max = max(q_table[next_state])
    # q_table[state][action] += alpha * (reward + gamma * next_max - old_value)

# Continuous simulation loop
# for message in consumer:

def get_node_names():
    try:
        # Load Kubernetes configuration (use load_incluster_config() if running inside a pod)
        config.load_kube_config()
        
        # Initialize CoreV1Api
        v1 = client.CoreV1Api()
        
        # List all nodes
        nodes = v1.list_node()
        
        # Extract node names
        node_names = [node.metadata.name for node in nodes.items]
        
        return node_names
    
    except client.ApiException as e:
        print(f"API exception: {e}")
        return []

# Example usage
# node_names = get_node_names()
# print(node_names)  # Output: ['kind-control-plane', 'kind-worker', 'kind-worker2']


if __name__ == "__main__":
    print("AC Agent started at ", datetime.datetime.now())
   
    # load kubernetes nodes
    nodes = get_node_names()
    print (nodes)

    try:
        for message in consumer:
            if message:
                print(f"Received message: {message.value}")
                nsr = message.value
                admission_control(nsr)


                # # Update time and manage slice expiration
                # current_time += 1
                # for slice_id, details in list(active_slices.items()):
                #     details["time_left"] -= 1
                #     if details["time_left"] <= 0:
                #         nodes[details["nodes"]["core"]]["CPU"] += details["resources"]["CPU_core"]
                #         nodes[details["nodes"]["core"]]["RAM"] += details["resources"]["RAM_core"]
                #         nodes[details["nodes"]["ran"]]["CPU"] += details["resources"]["CPU_ran"]
                #         nodes[details["nodes"]["ran"]]["RAM"] += details["resources"]["RAM_ran"]
                #         nodes[details["nodes"]["ran"]]["BW"] += details["resources"]["BW"]
                #         del active_slices[slice_id]
                #         print(f"Slice NSR_{slice_id} expired at time {current_time:.1f}s")

                # # Print current state every 5 seconds
                # if current_time % 5 == 0:
                #     state = get_state()
                #     print(f"\nTime: {current_time:.1f}s | Active slices: {len(active_slices)}")
                #     print(f"Remaining resources: {nodes}")
                #     print(f"State: {state}")
                #     print(f"Q-table size: {len(q_table)}")
                #     print("-" * 50)

                # time.sleep(1)  # Simulate 1 second per step
                
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        consumer.close()