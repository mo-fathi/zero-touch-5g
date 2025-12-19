# env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import heapq

class AdmissionEnv(gym.Env):
    def __init__(
        self,
        queue_length=20,
        total_resources={'cpu': 1000.0, 'mem': 10000.0, 'bw': 10000.0},
        penalty=-10.0,
        reject_penalty=-0.1,
        arrival_rate=0.1
    ):
        super().__init__()
        self.queue_length = queue_length
        self.total_resources = {k: float(v) for k, v in total_resources.items()}
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
        revenue = T0 * random.uniform(1, 20)

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
        return np.array([
            q['L_max_int'], q['L_max_ext'], q['Phi_min_int'], q['Phi_min_ext'],
            q['P_max_int'], q['P_max_ext'], nsr['T0'], nsr['revenue']
        ], dtype=np.float32)

    def get_remaining_res(self):
        return np.array([
            self.remaining_resources['cpu'],
            self.remaining_resources['mem'],
            self.remaining_resources['bw']
        ], dtype=np.float32)

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
        revenue_gained = 0.0
        reward = 0.0

        if action == 0:  # Reject
            reward = self.reject_penalty
        else:  # Accept
            req = self.current_nsr['resources']
            enough = all(self.remaining_resources[k] >= req[k] for k in req)
            if enough:
                for k in req:
                    self.remaining_resources[k] -= req[k]
                reward = self.current_nsr['revenue'] / self.current_nsr['T0']
                revenue_gained = self.current_nsr['revenue']
                release_time = self.current_time + self.current_nsr['T0']
                heapq.heappush(self.release_heap, (release_time, req.copy()))
            else:
                reward = self.penalty

        self.current_nsr = self._dequeue_next()
        done = self.current_nsr is None
        next_state = np.zeros(19, dtype=np.float32) if done else (
            np.concatenate((
                self.get_nsr_features(self.current_nsr),
                self.get_remaining_res(),
                [self.get_avg_queue()[0], self.get_avg_queue()[1]],
                self.get_avg_queue()[2]
            ))
        )

        return next_state, reward, done, False, {"revenue_gained": revenue_gained}