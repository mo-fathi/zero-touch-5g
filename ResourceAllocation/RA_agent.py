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
MAX_SLICES = 10               # Increase later for real use
FEATURE_DIM = 22              # Expanded for bandwidth and full QoS parameters
# ------------------------------------------------------------------

class SimpleClusterSimulator:
    def __init__(self):
        self.total_capacity_cpu = 200.0      # cores
        self.total_capacity_mem = 1024.0     # GiB
        self.total_capacity_bw = 1000.0      # MHz (added for radio bandwidth)
        self.active_slices: list[dict] = []

    def reset(self):
        self.active_slices.clear()

    def add_slice(self):
        if len(self.active_slices) >= MAX_SLICES:
            return
        num_nfs = random.randint(3, 6)  # Each slice has 3-6 network functions
        nfs = []
        for _ in range(num_nfs):
            nfs.append({
                "cpu_load": random.uniform(0.5, 3.0),
                "mem_load": random.uniform(2.0, 16.0),
            })
        self.active_slices.append({
            "nfs": nfs,  # List of network functions with their loads
            "required_cpu": random.uniform(2.0, 10.0),
            "required_mem": random.uniform(8.0, 64.0),
            "required_bw": random.uniform(50.0, 200.0),  # Added bandwidth requirement
            "target_int_latency_ms": random.uniform(5.0, 20.0),
            "target_int_loss": random.uniform(0.001, 0.01),
            "target_int_throughput": random.uniform(100.0, 500.0),  # Mbps
            "target_ext_latency_ms": random.uniform(10.0, 50.0),
            "target_ext_loss": random.uniform(0.001, 0.01),
            "target_ext_throughput": random.uniform(50.0, 200.0),  # Mbps
            "allocated_cpu": random.uniform(5.0, 15.0),
            "allocated_mem": random.uniform(16.0, 128.0),
            "allocated_bw": random.uniform(100.0, 300.0),  # Added
            "load_bw": random.uniform(50.0, 200.0),  # Added bandwidth load
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

    def apply_delta(self, idx: int, delta_cpu: float, delta_mem: float, delta_bw: float):
        if 0 <= idx < len(self.active_slices):
            s = self.active_slices[idx]
            s["allocated_cpu"] = np.clip(s["allocated_cpu"] + delta_cpu, s["required_cpu"] * 0.8, self.total_capacity_cpu)
            s["allocated_mem"] = np.clip(s["allocated_mem"] + delta_mem, s["required_mem"] * 0.8, self.total_capacity_mem)
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

        self.action_space = Box(low=-5.0, high=5.0, shape=(3 * MAX_SLICES,), dtype=np.float32)  # cpu, mem, bw deltas

        self.observation_space = GymDict({
            "per_slice": Box(low=0.0, high=1e6, shape=(MAX_SLICES, FEATURE_DIM), dtype=np.float32),
            "mask": Box(low=0, high=1, shape=(MAX_SLICES,), dtype=np.float32),
            "cluster": Box(low=0.0, high=1e6, shape=(6,), dtype=np.float32),  # Added bw cap/used
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

            # Mask padded actions
            mask_vec = np.zeros(MAX_SLICES)
            mask_vec[:active] = 1.0
            masked_action = action.copy()
            masked_action[:MAX_SLICES] *= mask_vec
            masked_action[MAX_SLICES:2*MAX_SLICES] *= mask_vec
            masked_action[2*MAX_SLICES:] *= mask_vec

            cpu_deltas = masked_action[:MAX_SLICES]
            mem_deltas = masked_action[MAX_SLICES:2*MAX_SLICES]
            bw_deltas = masked_action[2*MAX_SLICES:]

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
        per_slice = np.zeros((MAX_SLICES, FEATURE_DIM), dtype=np.float32)
        mask = np.zeros(MAX_SLICES, dtype=np.float32)
        cluster = np.zeros(6, dtype=np.float32)

        if self.simulate:
            active = len(self.simulator.active_slices)
            mask[:active] = 1.0

            used_cpu = used_mem = used_bw = 0.0

            for i in range(active):
                s = self.simulator.active_slices[i]
                load_cpu = sum(nf["cpu_load"] for nf in s["nfs"])  # Total from NFs
                load_mem = sum(nf["mem_load"] for nf in s["nfs"])  # Total from NFs
                usage_cpu = min(load_cpu, s["allocated_cpu"])
                usage_mem = min(load_mem, s["allocated_mem"])
                usage_bw = min(s["load_bw"], s["allocated_bw"])

                # Internal QoS (depends on CPU/mem)
                int_latency = s["target_int_latency_ms"] * (load_cpu / max(s["allocated_cpu"], 0.1))
                int_loss = max(0.0, load_cpu - s["allocated_cpu"]) / max(load_cpu, 0.1)
                int_throughput = s["allocated_cpu"] * 10.0  # Arbitrary model

                # External QoS (depends on bandwidth + CPU)
                ext_latency = s["target_ext_latency_ms"] * (s["load_bw"] / max(s["allocated_bw"], 0.1)) * (load_cpu / max(s["allocated_cpu"], 0.1))
                ext_loss = max(0.0, s["load_bw"] - s["allocated_bw"]) / max(s["load_bw"], 0.1)
                ext_throughput = s["allocated_bw"] * 5.0  # Arbitrary model

                per_slice[i] = np.array([
                    s["required_cpu"],
                    s["required_mem"],
                    s["required_bw"],
                    s["allocated_cpu"],
                    s["allocated_mem"],
                    s["allocated_bw"],
                    load_cpu,
                    load_mem,
                    s["load_bw"],
                    len(s["nfs"]),  # Number of network functions
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

                used_cpu += usage_cpu
                used_mem += usage_mem
                used_bw += usage_bw

            cluster = np.array([
                self.simulator.total_capacity_cpu,
                used_cpu,
                self.simulator.total_capacity_mem,
                used_mem,
                self.simulator.total_capacity_bw,
                used_bw,
            ], dtype=np.float32)

        return {"per_slice": per_slice, "mask": mask, "cluster": cluster}

    def _compute_reward(self) -> float:
        reward = 0.0
        total_used_cpu = total_allocated_cpu = total_used_mem = total_allocated_mem = total_used_bw = total_allocated_bw = 0.0

        for s in self.simulator.active_slices:
            load_cpu = sum(nf["cpu_load"] for nf in s["nfs"])
            load_mem = sum(nf["mem_load"] for nf in s["nfs"])

            # Compute current QoS (same as in observation)
            int_latency = s["target_int_latency_ms"] * (load_cpu / max(s["allocated_cpu"], 0.1))
            int_loss = max(0.0, load_cpu - s["allocated_cpu"]) / max(load_cpu, 0.1)
            int_throughput = s["allocated_cpu"] * 10.0

            ext_latency = s["target_ext_latency_ms"] * (s["load_bw"] / max(s["allocated_bw"], 0.1)) * (load_cpu / max(s["allocated_cpu"], 0.1))
            ext_loss = max(0.0, s["load_bw"] - s["allocated_bw"]) / max(s["load_bw"], 0.1)
            ext_throughput = s["allocated_bw"] * 5.0

            # QoS checks
            int_lat_ok = int_latency <= s["target_int_latency_ms"] * 1.1
            int_loss_ok = int_loss <= s["target_int_loss"] * 1.1
            int_thr_ok = int_throughput >= s["target_int_throughput"] * 0.9
            ext_lat_ok = ext_latency <= s["target_ext_latency_ms"] * 1.1
            ext_loss_ok = ext_loss <= s["target_ext_loss"] * 1.1
            ext_thr_ok = ext_throughput >= s["target_ext_throughput"] * 0.9

            qos_ok = all([int_lat_ok, int_loss_ok, int_thr_ok, ext_lat_ok, ext_loss_ok, ext_thr_ok])

            cpu_violation = max(0.0, load_cpu - s["allocated_cpu"])
            mem_violation = max(0.0, load_mem - s["allocated_mem"])
            bw_violation = max(0.0, s["load_bw"] - s["allocated_bw"])
            over_cpu = max(0.0, s["allocated_cpu"] - load_cpu * 1.4)
            over_mem = max(0.0, s["allocated_mem"] - load_mem * 1.4)
            over_bw = max(0.0, s["allocated_bw"] - s["load_bw"] * 1.4)

            if qos_ok:
                reward += 25.0
            else:
                reward -= 40.0

            # Violation penalties for each QoS
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

            reward -= 10.0 * cpu_violation
            reward -= 2.0 * mem_violation
            reward -= 5.0 * bw_violation
            reward -= 0.6 * over_cpu
            reward -= 0.3 * over_mem
            reward -= 0.4 * over_bw

            total_used_cpu += min(load_cpu, s["allocated_cpu"])
            total_allocated_cpu += s["allocated_cpu"]
            total_used_mem += min(load_mem, s["allocated_mem"])
            total_allocated_mem += s["allocated_mem"]
            total_used_bw += min(s["load_bw"], s["allocated_bw"])
            total_allocated_bw += s["allocated_bw"]

        # Efficiency bonuses for utilization
        if total_allocated_cpu > 0:
            cpu_util = total_used_cpu / total_allocated_cpu
            reward += 20.0 * cpu_util
        if total_allocated_mem > 0:
            mem_util = total_used_mem / total_allocated_mem
            reward += 10.0 * mem_util
        if total_allocated_bw > 0:
            bw_util = total_used_bw / total_allocated_bw
            reward += 15.0 * bw_util

        # Small costs for holding resources
        reward -= 0.03 * total_allocated_cpu
        reward -= 0.01 * total_allocated_mem
        reward -= 0.02 * total_allocated_bw

        return reward


class MaskedFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=256)

        self.slice_net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.cluster_net = nn.Sequential(
            nn.Linear(6, 64),  # Updated for bw
            nn.ReLU(),
        )
        self.final = nn.Linear(128 + 64, 256)

    def forward(self, observations) -> th.Tensor:
        slice_feat = observations["per_slice"]          # (B, MAX, F)
        mask = observations["mask"].unsqueeze(-1)       # (B, MAX, 1)

        encoded = self.slice_net(slice_feat)            # (B, MAX, 128)
        masked = encoded * mask

        # Mean pooling over active slices
        sum_pooled = masked.sum(dim=1)                  # (B, 128)
        active_count = mask.sum(dim=1).clamp(min=1.0)   # (B, 1)
        pooled = sum_pooled / active_count

        cluster_encoded = self.cluster_net(observations["cluster"])

        combined = th.cat([pooled, cluster_encoded], dim=1)
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
        learning_rate=3e-4,
        buffer_size=500_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
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