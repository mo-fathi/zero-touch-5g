# ac_agent_dqn.py
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import redis
import time
import random
import subprocess
import threading
import logging
from kafka import KafkaConsumer
from collections import deque
from kubernetes import client, config
from datetime import datetime, timedelta

# ====================== CONFIG ======================
KAFKA_BOOTSTRAP = ['localhost:9092']
KAFKA_TOPIC = 'lambda'
REDIS_HOST = 'localhost'
OAI_CORE_CHART = './charts/oai-5g-core/oai-5g-basic'
OAI_GNB_CHART = './charts/oai-5g-ran/oai-gnb'
OAI_UE_CHART = './charts/oai-5g-ran/oai-nr-ue'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DQN-AC-Agent")

# ====================== K8s & Redis & Kafka ======================
try:
    config.load_incluster_config()
except:
    config.load_kube_config()
v1 = client.CoreV1Api()

keeper = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP,
    auto_offset_reset='latest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# ====================== DQN ======================
class DQN(nn.Module):
    def __init__(self, state_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear( , 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(eval_net.state_dict())
optimizer = optim.Adam(eval_net.parameters(), lr=0.001)
buffer = deque(maxlen=10000)
epsilon = 1.0
BATCH_SIZE = 64
TARGET_UPDATE = 100
step = 0

# ====================== Active Slices Tracker ======================
active_slices = {}  # nsr_id → {'end_time': timestamp, 'namespace': str, 'revenue': int}
slice_lock = threading.Lock()

def cleanup_expired_slices():
    while True:
        now = time.time()
        with slice_lock:
            expired = [sid for sid, data in active_slices.items() if now > data['end_time']]
            for sid in expired:
                ns = active_slices[sid]['namespace']
                revenue = active_slices[sid]['revenue']
                try:
                    subprocess.run(["kubectl", "delete", "namespace", ns, "--grace-period=30"], timeout=60, check=True)
                    logger.info(f"SLICE TERMINATED: NSR_{sid} | +{revenue} revenue collected")
                except Exception as e:
                    logger.error(f"Failed to delete namespace {ns}: {e}")
                active_slices.pop(sid, None)
        time.sleep(10)

threading.Thread(target=cleanup_expired_slices, daemon=True).start()

# ====================== Resource Cache (every 5s) ======================
resource_cache = {'free_cpu': 0.0, 'free_mem': 0.0, 'last_update': 0}
cache_lock = threading.Lock()

def parse_quantity(q):
    """
    Parse Kubernetes resource strings → float (cores for CPU, bytes for memory)
    Supports: '100m', '2', '161748952n', '2Ki', '5Mi', '3Gi'
    """
    if q is None:
        return 0.0
    q = str(q).strip()
    if not q:
        return 0.0

    try:
        if q.endswith('n'):  # nanocores
            return float(q[:-1]) / 1_000_000_000.0
        elif q.endswith('u'):  # microcores
            return float(q[:-1]) / 1_000_000.0
        elif q.endswith('m'):  # millicores
            return float(q[:-1]) / 1000.0
        elif q.endswith('Ki'):
            return float(q[:-2]) * 1024
        elif q.endswith('Mi'):
            return float(q[:-2]) * 1024**2
        elif q.endswith('Gi'):
            return float(q[:-2]) * 1024**3
        elif q.endswith('Ti'):
            return float(q[:-2]) * 1024**4
        else:
            return float(q)  # plain number (cores or bytes)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse quantity '{q}': {e}")
        return 0.0

def update_resources():
    global resource_cache
    while True:
        try:
            nodes = v1.list_node().items
            metrics = client.CustomObjectsApi().list_cluster_custom_object("metrics.k8s.io", "v1beta1", "nodes")["items"]
            usage = {m["metadata"]["name"]: m["usage"] for m in metrics}

            total_alloc_cpu = total_used_cpu = 0.0
            total_alloc_mem = total_used_mem = 0.0

            for node in nodes:
                if any(role in (node.metadata.labels or {}) for role in ["control-plane", "master"]):
                    continue
                a = node.status.allocatable
                total_alloc_cpu += parse_quantity(a.get("cpu", 0))
                total_alloc_mem += parse_quantity(a.get("memory", 0))
                name = node.metadata.name
                if name in usage:
                    total_used_cpu += parse_quantity(usage[name]["cpu"])
                    total_used_mem += parse_quantity(usage[name]["memory"])

            free_cpu = max(0, total_alloc_cpu - total_used_cpu)
            free_mem = max(0, total_alloc_mem - total_used_mem)

            with cache_lock:
                resource_cache.update({
                    'free_cpu': free_cpu,
                    'free_mem': free_mem,
                    'last_update': time.time()
                })
        except Exception as e:
            logger.error(f"Resource update failed: {e}")
        time.sleep(5)

threading.Thread(target=update_resources, daemon=True).start()

# ====================== Deploy Slice with Helm ======================
def deploy_nsr(nsr):
    nsr_id = nsr["id"]
    namespace = f"nsr-{nsr_id}"
    t0 = nsr.get("T0", 3600)

    try:
        # 1. Create namespace
        subprocess.run(["kubectl", "create", "namespace", namespace], check=True, timeout=30)
        logger.info(f"Namespace {namespace} created")

        # 2. Deploy Core
        subprocess.run([
            "helm", "install", "oai-core", OAI_CORE_CHART,
            "--namespace", namespace, "--wait", "--timeout=5m"
        ], check=True, timeout=300)
        logger.info("OAI Core deployed")

        # 3. Deploy gNB
        subprocess.run([
            "helm", "install", "oai-gnb", OAI_GNB_CHART,
            "--namespace", namespace, "--wait", "--timeout=5m"
        ], check=True, timeout=300)
        logger.info("OAI gNB deployed")

        # 4. Deploy UE
        subprocess.run([
            "helm", "install", "oai-ue", OAI_UE_CHART,
            "--namespace", namespace, "--wait", "--timeout=5m"
        ], check=True, timeout=300)
        logger.info("OAI UE deployed")

        # 5. Record for auto-termination
        with slice_lock:
            active_slices[nsr_id] = {
                'end_time': time.time() + t0,
                'namespace': namespace,
                'revenue': nsr["revenue"]
            }

        logger.info(f"SLICE DEPLOYED: NSR_{nsr_id} | T0={t0}s | Rev={nsr['revenue']}")
        return True

    except Exception as e:
        logger.error(f"Deploy failed for NSR_{nsr_id}: {e}")
        try:
            subprocess.run(["kubectl", "delete", "namespace", namespace], timeout=30)
        except:
            pass
        return False

# ====================== State Builder ======================
def get_state(nsr):
    with cache_lock:
        free_cpu = resource_cache['free_cpu']
        free_mem = resource_cache['free_mem']

    q_size = int(keeper.get('q_size') or 0)
    sum_rev = float(keeper.get('sum_revenue') or 0)
    avg_rev = sum_rev / q_size if q_size > 0 else 0

    if not nsr:
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
        free_cpu / 100,           # max 100 cores
        free_mem / (50 * 1024**3), # max 50 GiB
        q_size / 50,
        avg_rev / 100,
        len(active_slices) / 20,  # active slices pressure
        *front
    ], dtype=np.float32)
    return state

# ====================== Main Loop ======================
logger.info("DQN 5G Admission Control Agent STARTED")
for message in consumer:
    nsr = message.value
    # global step
    step += 1

    # === 1. State ===
    state = get_state(nsr)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    # === 2. Action ===
    if random.random() < epsilon:
        action = random.randint(0, 1)
    else:
        with torch.no_grad():
            action = eval_net(state_t).argmax().item()

    # === 3. Demand Estimation ===
    qos = nsr["QoS"]
    demand_cpu = qos["Phi_min_int"] * 0.05 + qos["L_max_int"] * 0.1
    demand_mem = qos["Phi_min_int"] * 100 * 1024**2

    # === 4. Execute ===
    reward = 0
    deployed = False

    if action == 1 and resource_cache['free_cpu'] >= demand_cpu and resource_cache['free_mem'] >= demand_mem:
        deployed = deploy_nsr(nsr)
        reward = nsr["revenue"] / (nsr["T0"] / 60) if deployed else -50
        logger.info(f"{'ACCEPT' if deployed else 'REJECT (deploy fail)'} → NSR_{nsr['id']}")
    else:
        reward = -0.1 * nsr["revenue"] if resource_cache['free_cpu'] > 10 else 0
        logger.info(f"REJECT → NSR_{nsr['id']} | Rev: {nsr['revenue']}")

    # === 5. Update Redis ===
    keeper.decr('q_size')
    keeper.decrby('sum_revenue', nsr['revenue'])

    # === 6. Next State ===
    next_state = get_state(None)

    # === 7. Learn ===
    buffer.append((state, action, reward, next_state, False))
    if len(buffer) > BATCH_SIZE:
        batch = random.sample(buffer, BATCH_SIZE)
        s, a, r, s_, _ = map(np.array, zip(*batch))
        s = torch.tensor(s).to(device)
        a = torch.tensor(a).unsqueeze(1).to(device)
        r = torch.tensor(r).to(device)
        s_ = torch.tensor(s_).to(device)

        with torch.no_grad():
            target = r + 0.99 * target_net(s_).max(1)[0]
        current = eval_net(s).gather(1, a).squeeze()
        loss = nn.MSELoss()(current, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # === 8. Target Net & Epsilon ===
    if step % TARGET_UPDATE == 0:
        target_net.load_state_dict(eval_net.state_dict())
        epsilon = max(0.01, epsilon * 0.995)

    logger.info(f"Step {step} | ε={epsilon:.3f} | Active Slices: {len(active_slices)} | Buffer: {len(buffer)}")