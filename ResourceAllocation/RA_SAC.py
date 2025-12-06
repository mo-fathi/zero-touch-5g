import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as GymDict

import torch as th
import torch.nn as nn

from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
    
    env = NetSliceEnv(
        max_slices=MAX_SLICES,
        nunf_num=NUM_NFS,
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