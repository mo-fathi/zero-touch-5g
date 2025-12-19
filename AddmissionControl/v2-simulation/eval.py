# evaluate.py
import numpy as np
import matplotlib.pyplot as plt
from env import AdmissionEnv
from agents import DQNAgent, GreedyAgent
import torch
from datetime import datetime

MODEL_PATH = "models/dqn_agent-2025-12-20T02-11-59.pth"

def evaluate_agent(env, agent, num_episodes=100, is_dqn=True):
    rewards, revenues, util_hist = [], [], []

    for ep in range(num_episodes):
        seed = 10000 + ep
        state, _ = env.reset(seed=seed)
        ep_reward = ep_revenue = 0.0
        step_utils = []

        done = False
        while not done:
            action = agent.act(state, epsilon=0.0) if is_dqn else agent.act(state)
            next_state, reward, done, _, info = env.step(action)

            ep_reward += reward
            ep_revenue += info["revenue_gained"]

            used = {k: env.total_resources[k] - env.remaining_resources[k] for k in ['cpu', 'mem', 'bw']}
            util = {k: used[k] / env.total_resources[k] * 100 for k in used}
            step_utils.append((util['cpu'], util['mem'], util['bw']))

            state = next_state

        rewards.append(ep_reward)
        revenues.append(ep_revenue)
        util_hist.append(np.mean(step_utils, axis=0) if step_utils else np.zeros(3))

    return np.array(rewards), np.array(revenues), np.array(util_hist)

if __name__ == "__main__":
    env = AdmissionEnv(queue_length=20, arrival_rate=0.1)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Load trained DQN
    dqn_agent = DQNAgent(state_size, action_size)
    try:
        dqn_agent.model.load_state_dict(torch.load(MODEL_PATH))
        dqn_agent.model.eval()
        print("Loaded trained DQN model.")
    except FileNotFoundError:
        print("Error: 'dqn_agent.pth' not found. Run train.py first.")
        exit()

    greedy_agent = GreedyAgent()

    print("Evaluating both agents...")
    dqn_rewards, dqn_revenues, dqn_utils = evaluate_agent(env, dqn_agent, num_episodes=100, is_dqn=True)
    greedy_rewards, greedy_revenues, greedy_utils = evaluate_agent(env, greedy_agent, num_episodes=100, is_dqn=False)

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0,0].plot(dqn_rewards, label='DQN', alpha=0.8)
    axs[0,0].plot(greedy_rewards, label='Greedy', alpha=0.8)
    axs[0,0].set_title('Total Reward per Episode')
    axs[0,0].set_xlabel('Episode')
    axs[0,0].legend(); axs[0,0].grid()

    axs[0,1].plot(dqn_revenues, label='DQN', alpha=0.8)
    axs[0,1].plot(greedy_revenues, label='Greedy', alpha=0.8)
    axs[0,1].set_title('Total Revenue Achieved')
    axs[0,1].set_xlabel('Episode')
    axs[0,1].legend(); axs[0,1].grid()

    resources = ['CPU', 'Memory', 'BW']
    for i, res in enumerate(resources):
        axs[1,0].plot(dqn_utils[:, i], label=f'DQN {res}' if i==0 else None)
        axs[1,0].plot(greedy_utils[:, i], '--', label=f'Greedy {res}' if i==0 else None)
    axs[1,0].set_title('Average Resource Utilization (%)')
    axs[1,0].set_xlabel('Episode')
    axs[1,0].legend(); axs[1,0].grid()

    means_dqn = dqn_utils.mean(axis=0)
    means_greedy = greedy_utils.mean(axis=0)
    x = np.arange(3)
    axs[1,1].bar(x - 0.15, means_dqn, width=0.3, label='DQN')
    axs[1,1].bar(x + 0.15, means_greedy, width=0.3, label='Greedy')
    axs[1,1].set_xticks(x)
    axs[1,1].set_xticklabels(resources)
    axs[1,1].set_title('Mean Resource Utilization')
    axs[1,1].legend(); axs[1,1].grid(axis='y')

    plt.tight_layout()
    timestamp = datetime.now().isoformat().replace(":", "-").split(".")[0]
    plt.savefig(f"graphs/dqn_vs_greedy_comparison-{timestamp}.png", dpi=200)
    # plt.show()

    print("\n=== Summary ===")
    print(f"DQN     | Reward: {dqn_rewards.mean():.2f} | Revenue: {dqn_revenues.mean():.2f}")
    print(f"Greedy  | Reward: {greedy_rewards.mean():.2f} | Revenue: {greedy_revenues.mean():.2f}")
    print(f"DQN Util    → CPU: {means_dqn[0]:.1f}%  Mem: {means_dqn[1]:.1f}%  BW: {means_dqn[2]:.1f}%")
    print(f"Greedy Util → CPU: {means_greedy[0]:.1f}%  Mem: {means_greedy[1]:.1f}%  BW: {means_greedy[2]:.1f}%")