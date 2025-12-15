import gymnasium as gym
import torch as th
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Assuming NSENV.py is in the same directory or importable
from NSENV import NetSliceEnv

# ----------------------------- CONFIG -----------------------------
MAX_SLICES = 10               # Increase later for real use
NUM_NFS = 8                   # Fixed number of network functions per slice
SLICE_FEATURE_DIM = 2         # (allocated BW and BW usage) per slice
NF_FEATURE_DIM = 8            # 2 (cpu,mem) * 4 (request, limit, min, usage)
QOS_PARAMS = 6                # QoS parameters
# ---------------------------------------------------------------------

class MaskedFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=256)

        self.nf_net = nn.Sequential(
            nn.Linear(NF_FEATURE_DIM, 64),
            nn.ReLU(),
        )
        self.qos_net = nn.Sequential(
            nn.Linear(QOS_PARAMS * 2, 64),
            nn.ReLU(),
        )
        self.slice_net = nn.Sequential(
            nn.Linear(SLICE_FEATURE_DIM, 32),
            nn.ReLU(),
        )
        self.cluster_net = nn.Sequential(
            nn.Linear(QOS_PARAMS, 64),
            nn.ReLU(),
        )
        self.final = nn.Linear(160 + 64, 256)  # 64(nf) + 64(qos) + 32(slice) = 160 pooled, +64 cluster

    def forward(self, observations) -> th.Tensor:
        # NF processing
        nf_feat = observations["nf_features"]  # (B, MAX_S, NF_NUM, NF_FEATURE_DIM)
        # Assume no nf_mask, all NFs present
        encoded_nf = self.nf_net(nf_feat)  # (B, MAX_S, NF_NUM, 64)
        pooled_nf = encoded_nf.mean(dim=2)  # (B, MAX_S, 64) mean over NFs

        # QoS processing
        qos_feat = observations["QoS"]  # (B, MAX_S, QOS_PARAMS*2)
        encoded_qos = self.qos_net(qos_feat)  # (B, MAX_S, 64)

        # Slice features
        slice_feat = observations["slice_features"]  # (B, MAX_S, SLICE_FEATURE_DIM)
        encoded_slice = self.slice_net(slice_feat)  # (B, MAX_S, 32)

        # Combine per slice
        combined_slice = th.cat([pooled_nf, encoded_qos, encoded_slice], dim=-1)  # (B, MAX_S, 160)

        # Pool over slices
        slice_mask = observations["mask"].unsqueeze(-1)  # (B, MAX_S, 1)
        masked_slice = combined_slice * slice_mask
        sum_pooled = masked_slice.sum(dim=1)  # (B, 160)
        active_count = slice_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = sum_pooled / active_count

        cluster_encoded = self.cluster_net(observations["cluster"])  # (B, 64)

        combined = th.cat([pooled, cluster_encoded], dim=1)  # (B, 224)
        return self.final(combined)

if __name__ == "__main__":
    # Create the environment
    env = NetSliceEnv(
        max_slices=MAX_SLICES,
        nf_num=NUM_NFS,
        slice_feature_dim=SLICE_FEATURE_DIM,
        nf_feature_dim=NF_FEATURE_DIM,
        qos_params=QOS_PARAMS,
    )

    # Load the trained model (replace 'path_to_your_model.zip' with the actual saved model path, e.g., 'sac_5g_slice_agent-2025-12-07T12-34-56.zip')
    model_path = "sac_5g_slice_agent-2025-12-14T22-35-30.zip"  # <-- Replace this with your actual model file path
    model = SAC.load(model_path, env=env)

    # Test the model for one episode (up to 1000 steps)
    obs, _ = env.reset()
    steps = []
    rewards = []
    active_slices_list = []
    cpu_utils = []
    mem_utils = []
    bw_utils = []
    qos_satisfaction_fractions = []

    step = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Collect data
        steps.append(step)
        rewards.append(reward)
        active_slices_list.append(info["active_slices"])

        # Utilizations from cluster obs
        cluster = obs["cluster"]
        cpu_util = cluster[1] / cluster[0] if cluster[0] > 0 else 0
        mem_util = cluster[3] / cluster[2] if cluster[2] > 0 else 0
        bw_util = cluster[5] / cluster[4] if cluster[4] > 0 else 0
        cpu_utils.append(cpu_util)
        mem_utils.append(mem_util)
        bw_utils.append(bw_util)

        # QoS satisfaction fraction
        qos_results = env.simulate_qos(env.simulator.active_slices)
        if len(qos_results) > 0:
            satisfied_count = sum(all(res["qos_satisfied"].values()) for res in qos_results)
            qos_fraction = satisfied_count / len(qos_results)
        else:
            qos_fraction = 0
        qos_satisfaction_fractions.append(qos_fraction)

        step += 1

    # Plot the diagrams
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Resource Allocation Agent Test Results')

    # Plot 1: Reward per step
    axs[0, 0].plot(steps, rewards, label='Reward')
    axs[0, 0].set_title('Reward Over Steps')
    axs[0, 0].set_xlabel('Steps')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot 2: Active slices over steps
    axs[0, 1].plot(steps, active_slices_list, label='Active Slices', color='orange')
    axs[0, 1].set_title('Active Slices Over Steps')
    axs[0, 1].set_xlabel('Steps')
    axs[0, 1].set_ylabel('Number of Active Slices')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot 3: Resource utilizations
    axs[1, 0].plot(steps, cpu_utils, label='CPU Util')
    axs[1, 0].plot(steps, mem_utils, label='Mem Util')
    axs[1, 0].plot(steps, bw_utils, label='BW Util')
    axs[1, 0].set_title('Resource Utilizations Over Steps')
    axs[1, 0].set_xlabel('Steps')
    axs[1, 0].set_ylabel('Utilization (fraction)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot 4: QoS satisfaction fraction
    axs[1, 1].plot(steps, qos_satisfaction_fractions, label='QoS Satisfaction Fraction', color='green')
    axs[1, 1].set_title('QoS Satisfaction Fraction Over Steps')
    axs[1, 1].set_xlabel('Steps')
    axs[1, 1].set_ylabel('Fraction of Satisfied Slices')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    # Save the image
    plt.savefig('test_results.png')
    plt.close()
    print("Test completed. Diagrams saved as 'test_results.png'.")