import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from NSENV import NetSliceEnv

# --------------------------------------------------------------
# 1. Load the model (change the filename if you saved it differently)
# --------------------------------------------------------------
env = NetSliceEnv()                     # same env class you used for training
model = SAC.load("sac_5g_slice_agent-2025-12-14T22-35-30.zip", env=env)   # or "sac_5g_final" etc.

# --------------------------------------------------------------
# 2. Evaluation loop
# --------------------------------------------------------------
num_episodes = 10
max_steps = 1000

episode_rewards = []
episode_cpu_utils = []
episode_mem_utils = []
episode_bw_utils = []
episode_qos_satisfied = []
episode_cpu_overprovision = []
episode_mem_overprovision = []
episode_bw_overprovision = []
episode_active_slices = []

for ep in range(num_episodes):
    obs, _ = env.reset()
    step = 0
    done = False

    rewards = []
    cpu_utils = []
    mem_utils = []
    bw_utils = []
    qos_ok_percent = []
    cpu_overprov_percent = []
    mem_overprov_percent = []
    bw_overprov_percent = []
    active_list = []

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ---- Extract metrics from the simulator (safe because simulate=True) ----
        total_cpu_usage = total_cpu_alloc = total_mem_usage = total_mem_alloc = total_bw_load = total_bw_alloc = 0.0
        total_qos_ok = total_cpu_over = total_mem_over = total_bw_over = 0.0
        active = len(env.simulator.active_slices)

        for s in env.simulator.active_slices:
            cpu_usage = sum(nf["cpu_usage"] for nf in s["nfs"])
            mem_usage = sum(nf["mem_usage"] for nf in s["nfs"])
            alloc_cpu = sum(nf["requested_cpu"] for nf in s["nfs"])
            alloc_mem = sum(nf["requested_mem"] for nf in s["nfs"])
            bw_usage = s["bw_usage"]
            alloc_bw = s["allocated_bw"]

            total_cpu_usage += cpu_usage
            total_cpu_alloc += alloc_cpu
            total_mem_usage += mem_usage
            total_mem_alloc += alloc_mem
            total_bw_load += bw_usage
            total_bw_alloc += alloc_bw

            # QoS check (same logic as in reward function)
            int_latency = s["target_int_latency_ms"] * (cpu_usage / max(alloc_cpu, 0.1))
            int_loss = max(0.0, cpu_usage - alloc_cpu) / max(cpu_usage, 0.1)
            int_throughput = alloc_cpu * 10.0

            ext_latency = s["target_ext_latency_ms"] * (bw_usage / max(alloc_bw, 0.1)) * (cpu_usage / max(alloc_cpu, 0.1))
            ext_loss = max(0.0, bw_usage - alloc_bw) / max(bw_usage, 0.1)
            ext_throughput = alloc_bw * 5.0

            int_lat_ok = int_latency <= s["target_int_latency_ms"] * 1.1
            int_loss_ok = int_loss <= s["target_int_loss"] * 1.1
            int_thr_ok = int_throughput >= s["target_int_throughput"] * 0.9
            ext_lat_ok = ext_latency <= s["target_ext_latency_ms"] * 1.1
            ext_loss_ok = ext_loss <= s["target_ext_loss"] * 1.1
            ext_thr_ok = ext_throughput >= s["target_ext_throughput"] * 0.9

            qos_ok = all([int_lat_ok, int_loss_ok, int_thr_ok, ext_lat_ok, ext_loss_ok, ext_thr_ok])
            total_qos_ok += 1 if qos_ok else 0

            # Over-provisioning (extra resources beyond 40% headroom)
            total_cpu_over += max(0.0, alloc_cpu - cpu_usage * 1.4)
            total_mem_over += max(0.0, alloc_mem - mem_usage * 1.4)
            total_bw_over += max(0.0, alloc_bw - bw_usage * 1.4)

        cpu_utilization = (total_cpu_usage / total_cpu_alloc * 100) if total_cpu_alloc > 0 else 0
        mem_utilization = (total_mem_usage / total_mem_alloc * 100) if total_mem_alloc > 0 else 0
        bw_utilization = (total_bw_load / total_bw_alloc * 100) if total_bw_alloc > 0 else 0
        qos_percent = (total_qos_ok / active * 100) if active > 0 else 100
        cpu_overprov = (total_cpu_over / total_cpu_alloc * 100) if total_cpu_alloc > 0 else 0
        mem_overprov = (total_mem_over / total_mem_alloc * 100) if total_mem_alloc > 0 else 0
        bw_overprov = (total_bw_over / total_bw_alloc * 100) if total_bw_alloc > 0 else 0

        rewards.append(reward)
        cpu_utils.append(cpu_utilization)
        mem_utils.append(mem_utilization)
        bw_utils.append(bw_utilization)
        qos_ok_percent.append(qos_percent)
        cpu_overprov_percent.append(cpu_overprov)
        mem_overprov_percent.append(mem_overprov)
        bw_overprov_percent.append(bw_overprov)
        active_list.append(active)

        step += 1

    # Pad shorter episodes with last value so we can average
    def pad(arr):
        return np.pad(arr, (0, max_steps - len(arr)), constant_values=arr[-1] if arr else 0)

    episode_rewards.append(np.cumsum(pad(rewards)))   # cumsum after pad
    episode_cpu_utils.append(pad(cpu_utils))
    episode_mem_utils.append(pad(mem_utils))
    episode_bw_utils.append(pad(bw_utils))
    episode_qos_satisfied.append(pad(qos_ok_percent))
    episode_cpu_overprovision.append(pad(cpu_overprov_percent))
    episode_mem_overprovision.append(pad(mem_overprov_percent))
    episode_bw_overprovision.append(pad(bw_overprov_percent))
    episode_active_slices.append(pad(active_list))

# --------------------------------------------------------------
# 3. Convert to numpy arrays for easy averaging
# --------------------------------------------------------------
rewards_arr = np.array(episode_rewards)                     # (ep, steps)
cpu_utils_arr = np.array(episode_cpu_utils)
mem_utils_arr = np.array(episode_mem_utils)
bw_utils_arr = np.array(episode_bw_utils)
qos_arr = np.array(episode_qos_satisfied)
cpu_over_arr = np.array(episode_cpu_overprovision)
mem_over_arr = np.array(episode_mem_overprovision)
bw_over_arr = np.array(episode_bw_overprovision)
active_arr = np.array(episode_active_slices)

mean_reward = rewards_arr.mean(axis=0)
std_reward = rewards_arr.std(axis=0)

mean_cpu_util = cpu_utils_arr.mean(axis=0)
std_cpu_util = cpu_utils_arr.std(axis=0)

mean_mem_util = mem_utils_arr.mean(axis=0)
mean_bw_util = bw_utils_arr.mean(axis=0)
mean_qos = qos_arr.mean(axis=0)
mean_cpu_over = cpu_over_arr.mean(axis=0)
mean_mem_over = mem_over_arr.mean(axis=0)
mean_bw_over = bw_over_arr.mean(axis=0)
mean_active = active_arr.mean(axis=0)

steps = np.arange(max_steps)

# --------------------------------------------------------------
# 4. Plot everything (expanded for new metrics)
# --------------------------------------------------------------
plt.figure(figsize=(18, 12))

plt.subplot(3, 3, 1)
plt.plot(steps, mean_reward, label="Cumulative Reward")
plt.fill_between(steps, mean_reward - std_reward, mean_reward + std_reward, alpha=0.3)
plt.title("Cumulative Reward")
plt.xlabel("Step")
plt.grid(True)

plt.subplot(3, 3, 2)
plt.plot(steps, mean_cpu_util, color="green")
plt.fill_between(steps, mean_cpu_util - std_cpu_util, mean_cpu_util + std_cpu_util, color="green", alpha=0.3)
plt.title("CPU Utilization %")
plt.ylim(0, 110)
plt.ylabel("%")
plt.grid(True)

plt.subplot(3, 3, 3)
plt.plot(steps, mean_mem_util, color="blue")
plt.title("Memory Utilization %")
plt.ylim(0, 110)
plt.ylabel("%")
plt.grid(True)

plt.subplot(3, 3, 4)
plt.plot(steps, mean_bw_util, color="cyan")
plt.title("Bandwidth Utilization %")
plt.ylim(0, 110)
plt.ylabel("%")
plt.grid(True)

plt.subplot(3, 3, 5)
plt.plot(steps, mean_qos, color="purple")
plt.title("% Slices Meeting All QoS")
plt.ylim(0, 110)
plt.ylabel("%")
plt.grid(True)

plt.subplot(3, 3, 6)
plt.plot(steps, mean_cpu_over, color="orange")
plt.title("CPU Over-provisioning %")
plt.ylabel("%")
plt.grid(True)

plt.subplot(3, 3, 7)
plt.plot(steps, mean_mem_over, color="red")
plt.title("Memory Over-provisioning %")
plt.ylabel("%")
plt.grid(True)

plt.subplot(3, 3, 8)
plt.plot(steps, mean_bw_over, color="brown")
plt.title("Bandwidth Over-provisioning %")
plt.ylabel("%")
plt.grid(True)

plt.subplot(3, 3, 9)
plt.plot(steps, mean_active, color="black")
plt.title("Active Slices")
plt.xlabel("Step")
plt.ylim(0, MAX_SLICES + 1)
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final average metrics
print(f"Final average metrics over {num_episodes} episodes:")
print(f"  Final cumulative reward       : {mean_reward[-1]:.1f} ± {std_reward[-1]:.1f}")
print(f"  Final CPU utilization         : {mean_cpu_util[-1]:.1f}%")
print(f"  Final memory utilization      : {mean_mem_util[-1]:.1f}%")
print(f"  Final bandwidth utilization   : {mean_bw_util[-1]:.1f}%")
print(f"  Final QoS satisfied           : {mean_qos[-1]:.1f}%")
print(f"  Final CPU over-provisioning   : {mean_cpu_over[-1]:.1f}%")
print(f"  Final memory over-provisioning: {mean_mem_over[-1]:.1f}%")
print(f"  Final bandwidth over-provisioning: {mean_bw_over[-1]:.1f}%")
print(f"  Average active slices         : {mean_active[-1]:.1f}")