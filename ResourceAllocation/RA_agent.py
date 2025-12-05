import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as GymDict   # ← explicit name to avoid any conflict
import numpy as np
import random
from typing import Any

import torch as th
import torch.nn as nn

from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ----------------------------- CONFIG -----------------------------
MAX_SLICES = 50               # Maximum number of deployed network slice.
NUM_NFS = 8                   # Fixed number of network functions per slice (changed from max/random)
SLICE_FEATURE_DIM = 22        # Same as before for slice-level + QoS
NF_FEATURE_DIM = 6            # Per NF: req_cpu, alloc_cpu, load_cpu, req_mem, alloc_mem, load_mem
# ------------------------------------------------------------------

class SimpleClusterSimulator:
    def __init__(self):
        self.total_capacity_cpu = 500.0      # cores
        self.total_capacity_mem = 4096.0     # GiB
        self.total_capacity_bw = 5000.0      # MHz
        self.active_slices: list[dict] = []

    def reset(self):
        self.active_slices.clear()

    def add_slice(self):
        if len(self.active_slices) >= MAX_SLICES:
            return
        num_nfs = NUM_NFS  # Fixed to exactly 8
        nfs = []
        # The req_cpu and req_mem are the minimum amount of resource that a Network Function needs to run.
        # They set at the creation time and will not change during the running. They're like resources.requests.

        # The alloc_cpu and alloc_mem are the current actual resources assigned to the Network Functions. They will change by Resource Allocation agent.
        for _ in range(num_nfs):
            req_cpu = random.uniform(0.5, 2.0)
            req_mem = random.uniform(2.0, 16.0)
            alloc_cpu = random.uniform(1.0, 4.0)
            alloc_mem = random.uniform(4.0, 32.0)
            load_cpu = random.uniform(0.5, 3.0)
            load_mem = random.uniform(2.0, 16.0)
            nfs.append({
                "required_cpu": req_cpu,
                "required_mem": req_mem,
                "allocated_cpu": alloc_cpu,
                "allocated_mem": alloc_mem,
                "cpu_load": load_cpu,
                "mem_load": load_mem,
            })
        self.active_slices.append({
            "nfs": nfs,
            "required_bw": random.uniform(50.0, 200.0),
            "target_int_latency_ms": random.uniform(5.0, 20.0),
            "target_int_loss": random.uniform(0.001, 0.01),
            "target_int_throughput": random.uniform(100.0, 500.0),  # Mbps
            "target_ext_latency_ms": random.uniform(10.0, 50.0),
            "target_ext_loss": random.uniform(0.001, 0.01),
            "target_ext_throughput": random.uniform(50.0, 200.0),  # Mbps
            "allocated_bw": random.uniform(100.0, 300.0),
            "load_bw": random.uniform(50.0, 200.0),
        })

    def remove_slice(self, idx: int | None = None):
        if not self.active_slices:
            return
        if idx is None:
            idx = random.randint(0, len(self.active_slices) - 1)
        del self.active_slices[idx]

    def update_loads(self):
        for s in self.active_slices:
            for nf in s["nfs"]:
                nf["cpu_load"] = np.clip(nf["cpu_load"] + random.gauss(0, 0.2), 0.2, 10.0)
                nf["mem_load"] = np.clip(nf["mem_load"] + random.gauss(0, 1.0), 1.0, 64.0)
            s["load_bw"] = np.clip(s["load_bw"] + random.gauss(0, 10.0), 20.0, 500.0)

    def apply_delta(self, idx: int, cpu_deltas: np.ndarray, mem_deltas: np.ndarray, delta_bw: float):
        if 0 <= idx < len(self.active_slices):
            s = self.active_slices[idx]
            num_nfs = len(s["nfs"])
            for j in range(num_nfs):
                nf = s["nfs"][j]
                nf["allocated_cpu"] = np.clip(nf["allocated_cpu"] + cpu_deltas[j], nf["required_cpu"] * 0.8, np.inf)
                nf["allocated_mem"] = np.clip(nf["allocated_mem"] + mem_deltas[j], nf["required_mem"] * 0.8, np.inf)
            s["allocated_bw"] = np.clip(s["allocated_bw"] + delta_bw, s["required_bw"] * 0.8, self.total_capacity_bw)

class K8sSliceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, simulate: bool = True):
        super().__init__()
        self.simulate = simulate
        self.current_step = 0

        if self.simulate:
            self.simulator = SimpleClusterSimulator()
        else:
            raise NotImplementedError("Real Kubernetes mode not implemented in this example")

        action_dim = MAX_SLICES * (2 * NUM_NFS + 1)  # cpu/mem deltas per NF + bw per slice
        self.action_space = Box(low=-5.0, high=5.0, shape=(action_dim,), dtype=np.float32)

        self.observation_space = GymDict({
            "slice_features": Box(low=0.0, high=1e6, shape=(MAX_SLICES, SLICE_FEATURE_DIM), dtype=np.float32),
            "nf_features": Box(low=0.0, high=1e6, shape=(MAX_SLICES, NUM_NFS, NF_FEATURE_DIM), dtype=np.float32),
            "nf_mask": Box(low=0, high=1, shape=(MAX_SLICES, NUM_NFS), dtype=np.float32),
            "mask": Box(low=0, high=1, shape=(MAX_SLICES,), dtype=np.float32),
            "cluster": Box(low=0.0, high=1e6, shape=(6,), dtype=np.float32),  # cpu/mem/bw cap/used
        })

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        if self.simulate:
            self.simulator.reset()
            initial_slices = random.randint(2, MAX_SLICES // 2 + 1)
            for _ in range(initial_slices):
                self.simulator.add_slice()
            self.simulator.update_loads()

        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        if self.simulate:
            active = len(self.simulator.active_slices)

            # Unpack and mask actions
            cpu_deltas = np.zeros((MAX_SLICES, NUM_NFS))
            mem_deltas = np.zeros((MAX_SLICES, NUM_NFS))
            bw_deltas = np.zeros(MAX_SLICES)

            idx = 0
            for i in range(MAX_SLICES):
                cpu_deltas[i] = action[idx:idx + NUM_NFS]
                idx += NUM_NFS
                mem_deltas[i] = action[idx:idx + NUM_NFS]
                idx += NUM_NFS
                bw_deltas[i] = action[idx]
                idx += 1

            mask_vec = np.zeros(MAX_SLICES)
            mask_vec[:active] = 1.0
            for i in range(MAX_SLICES):
                if mask_vec[i] == 0:
                    cpu_deltas[i] = 0
                    mem_deltas[i] = 0
                    bw_deltas[i] = 0

            for i in range(active):
                self.simulator.apply_delta(i, cpu_deltas[i], mem_deltas[i], bw_deltas[i])

            # Dynamic arrival/departure
            if random.random() < 0.07 and active < MAX_SLICES:
                self.simulator.add_slice()
            if random.random() < 0.04 and active > 1:
                self.simulator.remove_slice()

            self.simulator.update_loads()

            reward = self._compute_reward()
            info["active_slices"] = len(self.simulator.active_slices)

            self.current_step += 1
            if self.current_step >= 1000:
                terminated = True

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> dict[str, np.ndarray]:
        slice_features = np.zeros((MAX_SLICES, SLICE_FEATURE_DIM), dtype=np.float32)
        nf_features = np.zeros((MAX_SLICES, NUM_NFS, NF_FEATURE_DIM), dtype=np.float32)
        nf_mask = np.zeros((MAX_SLICES, NUM_NFS), dtype=np.float32)
        mask = np.zeros(MAX_SLICES, dtype=np.float32)
        cluster = np.zeros(6, dtype=np.float32)

        if self.simulate:
            active = len(self.simulator.active_slices)
            mask[:active] = 1.0

            used_cpu = used_mem = used_bw = 0.0

            for i in range(active):
                s = self.simulator.active_slices[i]
                num_nfs = len(s["nfs"])
                nf_mask[i, :num_nfs] = 1.0

                total_req_cpu = total_alloc_cpu = total_load_cpu = 0.0
                total_req_mem = total_alloc_mem = total_load_mem = 0.0

                for j in range(num_nfs):
                    nf = s["nfs"][j]
                    nf_features[i, j] = np.array([
                        nf["required_cpu"],
                        nf["allocated_cpu"],
                        nf["cpu_load"],
                        nf["required_mem"],
                        nf["allocated_mem"],
                        nf["mem_load"],
                    ], dtype=np.float32)

                    total_req_cpu += nf["required_cpu"]
                    total_alloc_cpu += nf["allocated_cpu"]
                    total_load_cpu += nf["cpu_load"]
                    total_req_mem += nf["required_mem"]
                    total_alloc_mem += nf["allocated_mem"]
                    total_load_mem += nf["mem_load"]

                    used_cpu += min(nf["cpu_load"], nf["allocated_cpu"])
                    used_mem += min(nf["mem_load"], nf["allocated_mem"])

                used_bw += min(s["load_bw"], s["allocated_bw"])

                # Internal QoS
                int_latency = s["target_int_latency_ms"] * (total_load_cpu / max(total_alloc_cpu, 0.1))
                int_loss = max(0.0, total_load_cpu - total_alloc_cpu) / max(total_load_cpu, 0.1)
                int_throughput = total_alloc_cpu * 10.0

                # External QoS
                ext_latency = s["target_ext_latency_ms"] * (s["load_bw"] / max(s["allocated_bw"], 0.1)) * (total_load_cpu / max(total_alloc_cpu, 0.1))
                ext_loss = max(0.0, s["load_bw"] - s["allocated_bw"]) / max(s["load_bw"], 0.1)
                ext_throughput = s["allocated_bw"] * 5.0

                slice_features[i] = np.array([
                    s["required_bw"],
                    s["allocated_bw"],
                    s["load_bw"],
                    total_req_cpu,
                    total_alloc_cpu,
                    total_load_cpu,
                    total_req_mem,
                    total_alloc_mem,
                    total_load_mem,
                    num_nfs,
                    int_latency,
                    int_loss,
                    int_throughput,
                    ext_latency,
                    ext_loss,
                    ext_throughput,
                    s["target_int_latency_ms"],
                    s["target_int_loss"],
                    s["target_int_throughput"],
                    s["target_ext_latency_ms"],
                    s["target_ext_loss"],
                    s["target_ext_throughput"],
                ], dtype=np.float32)

            cluster = np.array([
                self.simulator.total_capacity_cpu,
                used_cpu,
                self.simulator.total_capacity_mem,
                used_mem,
                self.simulator.total_capacity_bw,
                used_bw,
            ], dtype=np.float32)

        return {
            "slice_features": slice_features,
            "nf_features": nf_features,
            "nf_mask": nf_mask,
            "mask": mask,
            "cluster": cluster
        }

    def _compute_reward(self) -> float:
        reward = 0.0
        total_used_cpu = total_allocated_cpu = total_used_mem = total_allocated_mem = total_used_bw = total_allocated_bw = 0.0
        total_cpu_violation = total_mem_violation = total_bw_violation = 0.0
        total_cpu_over = total_mem_over = total_bw_over = 0.0

        for s in self.simulator.active_slices:
            total_load_cpu = total_alloc_cpu = total_load_mem = total_alloc_mem = 0.0
            cpu_viol = mem_viol = 0.0
            cpu_ov = mem_ov = 0.0

            for nf in s["nfs"]:
                cpu_viol += max(0.0, nf["cpu_load"] - nf["allocated_cpu"])
                mem_viol += max(0.0, nf["mem_load"] - nf["allocated_mem"])
                cpu_ov += max(0.0, nf["allocated_cpu"] - nf["cpu_load"] * 1.4)
                mem_ov += max(0.0, nf["allocated_mem"] - nf["mem_load"] * 1.4)

                total_load_cpu += nf["cpu_load"]
                total_alloc_cpu += nf["allocated_cpu"]
                total_load_mem += nf["mem_load"]
                total_alloc_mem += nf["allocated_mem"]

                total_used_cpu += min(nf["cpu_load"], nf["allocated_cpu"])
                total_allocated_cpu += nf["allocated_cpu"]
                total_used_mem += min(nf["mem_load"], nf["allocated_mem"])
                total_allocated_mem += nf["allocated_mem"]

            bw_viol = max(0.0, s["load_bw"] - s["allocated_bw"])
            bw_ov = max(0.0, s["allocated_bw"] - s["load_bw"] * 1.4)

            total_bw_violation += bw_viol
            total_bw_over += bw_ov
            total_used_bw += min(s["load_bw"], s["allocated_bw"])
            total_allocated_bw += s["allocated_bw"]

            # QoS (slice-level)
            int_latency = s["target_int_latency_ms"] * (total_load_cpu / max(total_alloc_cpu, 0.1))
            int_loss = max(0.0, total_load_cpu - total_alloc_cpu) / max(total_load_cpu, 0.1)
            int_throughput = total_alloc_cpu * 10.0

            ext_latency = s["target_ext_latency_ms"] * (s["load_bw"] / max(s["allocated_bw"], 0.1)) * (total_load_cpu / max(total_alloc_cpu, 0.1))
            ext_loss = max(0.0, s["load_bw"] - s["allocated_bw"]) / max(s["load_bw"], 0.1)
            ext_throughput = s["allocated_bw"] * 5.0

            int_lat_ok = int_latency <= s["target_int_latency_ms"] * 1.1
            int_loss_ok = int_loss <= s["target_int_loss"] * 1.1
            int_thr_ok = int_throughput >= s["target_int_throughput"] * 0.9
            ext_lat_ok = ext_latency <= s["target_ext_latency_ms"] * 1.1
            ext_loss_ok = ext_loss <= s["target_ext_loss"] * 1.1
            ext_thr_ok = ext_throughput >= s["target_ext_throughput"] * 0.9

            qos_ok = all([int_lat_ok, int_loss_ok, int_thr_ok, ext_lat_ok, ext_loss_ok, ext_thr_ok])

            if qos_ok:
                reward += 25.0
            else:
                reward -= 40.0

            # Specific QoS penalties
            if not int_lat_ok:
                reward -= 10.0 * (int_latency - s["target_int_latency_ms"] * 1.1) / s["target_int_latency_ms"]
            if not int_loss_ok:
                reward -= 5.0 * (int_loss - s["target_int_loss"] * 1.1) / s["target_int_loss"]
            if not int_thr_ok:
                reward -= 10.0 * (s["target_int_throughput"] * 0.9 - int_throughput) / s["target_int_throughput"]
            if not ext_lat_ok:
                reward -= 10.0 * (ext_latency - s["target_ext_latency_ms"] * 1.1) / s["target_ext_latency_ms"]
            if not ext_loss_ok:
                reward -= 5.0 * (ext_loss - s["target_ext_loss"] * 1.1) / s["target_ext_loss"]
            if not ext_thr_ok:
                reward -= 10.0 * (s["target_ext_throughput"] * 0.9 - ext_throughput) / s["target_ext_throughput"]

            reward -= 10.0 * cpu_viol
            reward -= 2.0 * mem_viol
            reward -= 5.0 * bw_viol
            reward -= 0.6 * cpu_ov
            reward -= 0.3 * mem_ov
            reward -= 0.4 * bw_ov

        # Global efficiency bonuses
        if total_allocated_cpu > 0:
            cpu_util = total_used_cpu / total_allocated_cpu
            reward += 20.0 * cpu_util
        if total_allocated_mem > 0:
            mem_util = total_used_mem / total_allocated_mem
            reward += 10.0 * mem_util
        if total_allocated_bw > 0:
            bw_util = total_used_bw / total_allocated_bw
            reward += 15.0 * bw_util

        # Global over-allocation penalties
        cpu_excess = max(0.0, total_allocated_cpu - self.simulator.total_capacity_cpu)
        mem_excess = max(0.0, total_allocated_mem - self.simulator.total_capacity_mem)
        bw_excess = max(0.0, total_allocated_bw - self.simulator.total_capacity_bw)
        reward -= 100.0 * (cpu_excess + mem_excess + bw_excess)

        # Small holding costs
        reward -= 0.03 * total_allocated_cpu
        reward -= 0.01 * total_allocated_mem
        reward -= 0.02 * total_allocated_bw

        return reward


class MaskedFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=256)

        self.nf_net = nn.Sequential(
            nn.Linear(NF_FEATURE_DIM, 64),
            nn.ReLU(),
        )
        self.slice_net = nn.Sequential(
            nn.Linear(SLICE_FEATURE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.cluster_net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
        )
        self.final = nn.Linear(192 + 64, 256)  # 128 (slice) + 64 (nf pool) = 192 per slice, then pooled

    def forward(self, observations) -> th.Tensor:
        # NF processing
        nf_feat = observations["nf_features"]  # (B, MAX_S, MAX_N, 6)
        nf_mask = observations["nf_mask"].unsqueeze(-1)  # (B, MAX_S, MAX_N, 1)

        encoded_nf = self.nf_net(nf_feat)  # (B, MAX_S, MAX_N, 64)
        masked_nf = encoded_nf * nf_mask
        sum_pooled_nf = masked_nf.sum(dim=2)  # (B, MAX_S, 64)
        active_nf_count = nf_mask.sum(dim=2).clamp(min=1.0)  # (B, MAX_S, 1)
        pooled_nf = sum_pooled_nf / active_nf_count  # (B, MAX_S, 64)

        # Slice features
        slice_feat = observations["slice_features"]  # (B, MAX_S, 22)
        encoded_slice = self.slice_net(slice_feat)  # (B, MAX_S, 128)

        # Combine per slice
        combined_slice = th.cat([encoded_slice, pooled_nf], dim=-1)  # (B, MAX_S, 192)

        # Pool over slices
        slice_mask = observations["mask"].unsqueeze(-1)  # (B, MAX_S, 1)
        masked_slice = combined_slice * slice_mask
        sum_pooled = masked_slice.sum(dim=1)  # (B, 192)
        active_count = slice_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = sum_pooled / active_count

        cluster_encoded = self.cluster_net(observations["cluster"])  # (B, 64)

        combined = th.cat([pooled, cluster_encoded], dim=1)  # (B, 256)
        return self.final(combined)


if __name__ == "__main__":
    env = K8sSliceEnv(simulate=True)

    policy_kwargs = dict(
        features_extractor_class=MaskedFeatureExtractor,
        features_extractor_kwargs=dict(),
        net_arch=[256, 256],
    )

    model = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        #  learning rate for adam optimizer, 
        #the same learning rate will be used for all networks (Q-Values, Actor and Value function) 
        #it can be a function of the current progress remaining (from 1 to 0)
        learning_rate=3e-4,
        buffer_size=500_000,
        learning_starts=5_000,
        # Minibatch size for each gradient update
        batch_size=256,
        #  the soft update coefficient (“Polyak update”, between 0 and 1)
        tau=0.005,
        # the discount factor
        gamma=0.99,
        #  Update the model every train_freq steps. Alternatively pass a tuple of frequency and unit like (5, "step") or (2, "episode").
        train_freq=1,
        # How many gradient steps to do after each rollout (see train_freq) Set to -1 means to do as many gradient steps as steps done in the environment during the rollout.
        gradient_steps=1,
        #  Entropy regularization coefficient. (Equivalent to inverse of reward scale in the original SAC paper.) Controlling exploration/exploitation trade-off. Set it to ‘auto’ to learn it automatically
        ent_coef="auto",
        verbose=1,
        tensorboard_log="./sac_5g_logs",
        device="auto",
    )

    print("Starting training...")
    model.learn(total_timesteps=400_000, progress_bar=True)
    model.save("sac_5g_slice_agent.zip")

    # Quick evaluation
    obs, _ = env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()