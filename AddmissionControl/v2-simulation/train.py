# train.py
from env import AdmissionEnv
from agents import DQNAgent
import torch
from datetime import datetime

if __name__ == "__main__":
    env = AdmissionEnv(queue_length=20, arrival_rate=0.1)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    episodes = 500
    batch_size = 32

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if (e + 1) % 100 == 0:
            print(f"Episode {e+1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    # Save the trained model
    timestamp = datetime.now().isoformat().replace(":", "-").split(".")[0]
    torch.save(agent.model.state_dict(), f"models/dqn_agent-{timestamp}.pth")
    print(f"Training complete. Model saved as 'dqn_agent-{timestamp}.pth'")