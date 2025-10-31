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
import redis
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
#import matplotlib.pyplot as plt


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

# Cluster configuration
# a list of node names 
nodes = []
    
# List of possible actions for Admission Control
AC_ACTIONS = ["Accept", "Reject"]


# Active slices tracking
active_slices = {}

# Q-learning parameters
q_table = {}  # State-action pairs
alpha = 0.1   # Learning rate
gamma = 0.9   # Discount factor
epsilon = 1.0 # Exploration rate


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

# Redis configuration
keeper = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True  # ensures returned values are strings instead of bytes
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

# TODO normalizing the state     
# Q-learning action selection
def choose_action(state):
    if state not in q_table:
        q_table[state] = [0, 0]  # Initialize for accept (1) and reject (0)
    
    if random.uniform(0, 1) < epsilon:
        return random.choose(AC_ACTIONS)  # Explore
    else:
        return ACTIONS[np.argmax(q_table[state])]  # Exploit
    
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


def reward_function():
    # SUM (Revenue - Cost) for all ns
    reward = 0.0
    for ns in networkSlices:
        reward += (get_ns_revenue() - get_ns_cost()) * T0

    return reward


def get_ns_cost(ns_id):
    # return current cost
    return 0
def get_ns_revenue(ns_id):
    return 0
    
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



## Admission Control State
# S_ac = {Remaining CPU & RAM for both core and ran nodes, 
#           NSR[i],
#           queue length,
#           revenue avg in queue
#           QoS avg in queue
#             }
def get_AC_state(nsr):

    # get cluster remaining resource state
    state = get_resource_metrics()

    # add nsr to state
    state['L_max_int'] = nsr["QoS"]["L_max_int"]
    state['L_max_ext'] = nsr["QoS"]["L_max_ext"]
    state['Phi_min_int'] = nsr["QoS"]["Phi_min_int"]
    state['Phi_min_ext'] = nsr["QoS"]["Phi_min_ext"]
    state['P_max_int'] = nsr["QoS"]["P_max_int"]
    state['P_max_ext'] = nsr["QoS"]["P_max_ext"]



    # add queue parameter to state
    state['n'] = keeper.get('q_size')
    state['revenue_avg'] = keeper.get('sum_of_revenue') / state['n']
    # TODO Add lambda QoS parameter to state


    return state


# AC Agent with Q-learning
def admission_control(nsr):

    # get RL state of environment
    state = get_AC_state(nsr)
    
    # Choose action by algorithm 
    action = choose_action(state)

    # Calculate Reward
    reward = reward_function()
    # print(action)
    
    
    if action == 1:
        # Publish accepted NSR to deploy topic
        print("nsr id ",nsr.get('id', 'default'), "  accepted")
        producer.send(producer_topic, nsr)
        producer.flush()  # Ensure message is sent immediately
        
    else:
        print("nsr id ",nsr.get('id', 'default'), "  rejected")


if __name__ == "__main__":
    print("AC Agent started at ", datetime.datetime.now())
   
    # load kubernetes nodes
    nodes = get_node_names()
    # print (nodes)

    # creating NNs
    action_dim = 2  # Accept/Reject
    # TODO
    # state_dim = len(get_AC_state())
    state_dim = 13

    eval_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)

    target_net.load_state_dict(eval_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(eval_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()


# Admission Control cycle:
    try:
        for message in consumer:
            if message:
                print(f"Received message: {message.value}")
                nsr = message.value
                admission_control(nsr)
                
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        consumer.close()