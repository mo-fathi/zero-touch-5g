import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

# --------------------------------------------------------------
# 1. Load the model (change the filename if you saved it differently)
# --------------------------------------------------------------
env = K8sSliceEnv(simulate=True)                     # same env class you used for training
model = SAC.load("sac_5g_slice_agent.zip", env=env)   # or "sac_5g_final" etc.

# --------------------------------------------------------------
# 2. Evaluation loop
# --------------------------------------------------------------
num_episodes = 10
max_steps = 1000

episode_rewards = []
episode_utilizations = []
episode_qos_satisfied = []
episode_overprovision = []
episode_active_slices = []

for ep in range(num_episodes):
    obs, _ = env.reset()
    step = 0
    done = False

    rewards = []
    utils = []
    qos_ok_percent = []
    overprov_percent = []
    active_list = []

    while not done and step < max_steps:   # ← FIXED: removed "is False"
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ---- Extract metrics from the simulator (safe because simulate=True) ----
        total_load = total_alloc = total_qos_ok = total_over = 0.0
        active = len(env.simulator.active_slices)

        for s in env.simulator.active_slices:
            total_load += s["load_cpu"]
            total_alloc += s["allocated_cpu"]

            # QoS check (same logic as in reward function)
            latency_ok = (s["load_cpu"] / max(s["allocated_cpu"], 0.1)) <= 1.1
            cpu_ok = s["load_cpu"] <= s["allocated_cpu"] + 0.5
            qos_ok = latency_ok and cpu_ok
            total_qos_ok += 1 if qos_ok else 0

            # Over-provisioning (extra resources beyond 40% headroom)
            total_over += max(0.0, s["allocated_cpu"] - s["load_cpu"] * 1.4)

        utilization = (total_load / total_alloc * 100) if total_alloc > 0 else 0
        qos_percent = (total_qos_ok / active * 100) if active > 0 else 100
        overprov = (total_over / total_alloc * 100) if total_alloc > 0 else 0

        rewards.append(reward)
        utils.append(utilization)
        qos_ok_percent.append(qos_percent)
        overprov_percent.append(overprov)
        active_list.append(active)

        step += 1

    # Pad shorter episodes with last value so we can average
    def pad(arr):
        return np.pad(arr, (0, max_steps - len(arr)), constant_values=arr[-1] if arr else 0)

    episode_rewards.append(np.cumsum(pad(rewards)))   # ← Also fixed: cumsum after pad to handle short eps
    episode_utilizations.append(pad(utils))
    episode_qos_satisfied.append(pad(qos_ok_percent))
    episode_overprovision.append(pad(overprov_percent))
    episode_active_slices.append(pad(active_list))

# --------------------------------------------------------------
# 3. Convert to numpy arrays for easy averaging
# --------------------------------------------------------------
rewards_arr = np.array(episode_rewards)                     # (ep, steps)
utils_arr = np.array(episode_utilizations)
qos_arr = np.array(episode_qos_satisfied)
over_arr = np.array(episode_overprovision)
active_arr = np.array(episode_active_slices)

mean_reward = rewards_arr.mean(axis=0)
std_reward = rewards_arr.std(axis=0)

mean_util = utils_arr.mean(axis=0)
std_util = utils_arr.std(axis=0)

mean_qos = qos_arr.mean(axis=0)
mean_over = over_arr.mean(axis=0)
mean_active = active_arr.mean(axis=0)

steps = np.arange(max_steps)

# --------------------------------------------------------------
# 4. Plot everything
# --------------------------------------------------------------
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(steps, mean_reward, label="Cumulative Reward")
plt.fill_between(steps, mean_reward - std_reward, mean_reward + std_reward, alpha=0.3)
plt.title("Cumulative Reward")
plt.xlabel("Step")
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(steps, mean_util, color="green")
plt.fill_between(steps, mean_util - std_util, mean_util + std_util, color="green", alpha=0.3)
plt.title("CPU Utilization %")
plt.ylim(0, 110)
plt.ylabel("%")
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(steps, mean_qos, color="purple")
plt.title("% Slices Meeting QoS")
plt.ylim(0, 110)
plt.ylabel("%")
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(steps, mean_over, color="orange")
plt.title("Over-provisioning %")
plt.ylabel("%")
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(steps, mean_active, color="red")
plt.title("Active Slices")
plt.xlabel("Step")
plt.ylim(0, MAX_SLICES + 1)
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final average metrics
print(f"Final average metrics over {num_episodes} episodes:")
print(f"  Final cumulative reward       : {mean_reward[-1]:.1f} ± {std_reward[-1]:.1f}")
print(f"  Final CPU utilization       : {mean_util[-1]:.1f}%")
print(f"  Final QoS satisfied        : {mean_qos[-1]:.1f}%")
print(f"  Final over-provisioning    : {mean_over[-1]:.1f}%")
print(f"  Average active slices        : {mean_active[-1]:.1f}")
