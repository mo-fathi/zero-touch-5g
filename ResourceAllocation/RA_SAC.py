import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as GymDict

import torch as th
import torch.nn as nn

from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

from NSENV import NetSliceEnv

from datetime import datetime
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

# Modifications to RA_SAC.py
# Add the following imports:
from stable_baselines3.common.callbacks import BaseCallback

# Add the following class after the model definition:
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.locals.get('infos') and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            self.logger.record('train/active_slices', info.get('active_slices', 0))
            self.logger.record('train/cluster_cpu_requested', info.get('cluster_cpu_requested', 0.0))
            self.logger.record('train/cluster_cpu_used', info.get('cluster_cpu_used', 0.0))
            self.logger.record('train/cluster_mem_requested', info.get('cluster_mem_requested', 0.0))
            self.logger.record('train/cluster_mem_used', info.get('cluster_mem_used', 0.0))
            self.logger.record('train/cluster_bw_allocated', info.get('cluster_bw_allocated', 0.0))
            self.logger.record('train/cluster_bw_used', info.get('cluster_bw_used', 0.0))
            self.logger.record('train/cpu_util', info.get('cpu_util', 0.0))
            self.logger.record('train/mem_util', info.get('mem_util', 0.0))
            self.logger.record('train/bw_util', info.get('bw_util', 0.0))
            self.logger.record('train/fraction_qos_ok', info.get('fraction_qos_ok', 0.0))
            self.logger.record('train/fraction_int_ok', info.get('fraction_int_ok', 0.0))
            self.logger.record('train/fraction_ext_ok', info.get('fraction_ext_ok', 0.0))
            self.logger.record('train/total_cpu_over', info.get('total_cpu_over', 0.0))
            self.logger.record('train/total_mem_over', info.get('total_mem_over', 0.0))
            self.logger.record('train/total_bw_over', info.get('total_bw_over', 0.0))
            self.logger.record('train/avg_int_latency', info.get('avg_int_latency', 0.0))
            self.logger.record('train/avg_int_loss', info.get('avg_int_loss', 0.0))
            self.logger.record('train/avg_int_throughput', info.get('avg_int_throughput', 0.0))
            self.logger.record('train/avg_ext_latency', info.get('avg_ext_latency', 0.0))
            self.logger.record('train/avg_ext_loss', info.get('avg_ext_loss', 0.0))
            self.logger.record('train/avg_ext_throughput', info.get('avg_ext_throughput', 0.0))
        return True

if __name__ == "__main__":
    
    env = NetSliceEnv(
        max_slices=MAX_SLICES,
        nf_num=NUM_NFS,
        slice_feature_dim=SLICE_FEATURE_DIM,
        nf_feature_dim=NF_FEATURE_DIM,
        qos_params=QOS_PARAMS,
    )

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
        buffer_size=125_000,
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

    timestamp = datetime.now().isoformat().replace(":", "-").split(".")[0]

    print(f"Starting training at {timestamp}...")
    
    model.learn(total_timesteps=10_000, progress_bar=True, callback=TensorboardCallback())
    model.save(f"sac_5g_slice_agent-{timestamp}.zip")

    # Quick evaluation
    obs, _ = env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()