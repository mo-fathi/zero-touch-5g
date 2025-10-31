import random
import numpy as np

# Actions
ACTIONS = ["Accept", "Reject"]

class AdmissionControlQLearning:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration rate
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # Q-values: key=state_str, value=[Q_accept, Q_reject]

    def state_to_key(self, state):
        """Convert dict state to a hashable string key for Q-table."""
        # Include remaining resources and NSR QoS in state key
        nsr = state["NSR"]
        qos = nsr["QoS"]
        key = (
            f"{state['remaining_cpu_cores']}_"
            f"{state['remaining_ram_cores']}_"
            f"{state['remaining_cpu_rans']}_"
            f"{state['remaining_ram_rans']}_"
            f"{state['remaining_bandwith_tans']}_"
            f"{qos['L_max_int']:.2f}_{qos['L_max_ext']:.2f}_"
            f"{qos['Phi_min_int']:.2f}_{qos['Phi_min_ext']:.2f}_"
            f"{qos['P_max_int']:.4f}_{qos['P_max_ext']:.4f}"
        )
        return key

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        key = self.state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = [0.0, 0.0]  # initialize Q-values
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        else:
            return ACTIONS[np.argmax(self.q_table[key])]

    def update_q(self, state, action, reward, next_state):
        """Q-Learning update."""
        key = self.state_to_key(state)
        next_key = self.state_to_key(next_state)
        if next_key not in self.q_table:
            self.q_table[next_key] = [0.0, 0.0]

        action_index = ACTIONS.index(action)
        best_next_q = max(self.q_table[next_key])
        self.q_table[key][action_index] += self.alpha * (
            reward + self.gamma * best_next_q - self.q_table[key][action_index]
        )

    def reward_function(self, state, action):
        """Reward for admission control."""
        nsr = state["NSR"]
        qos = nsr["QoS"]

        if action == "Reject":
            # small negative reward for rejecting (discourage unnecessary rejection)
            return -0.1

        # If Accept, check if resources are sufficient
        cpu_needed = nsr["QoS"]["Phi_min_int"] / 50  # example conversion
        ram_needed = nsr["QoS"]["Phi_min_ext"] / 50
        bw_needed = (nsr["QoS"]["Phi_min_int"] + nsr["QoS"]["Phi_min_ext"]) / 2

        if (
            state["remaining_cpu_cores"] >= cpu_needed
            and state["remaining_ram_cores"] >= ram_needed
            and state["remaining_cpu_rans"] >= cpu_needed
            and state["remaining_ram_rans"] >= ram_needed
            and state["remaining_bandwith_tans"] >= bw_needed
        ):
            # positive reward for accepting feasible NSR
            return 1.0
        else:
            # negative reward if accepting NSR exceeds resources (SLA violation)
            return -1.0

    def apply_nsr(self, state, nsr):
        """Update remaining resources after accepting an NSR."""
        cpu_needed = nsr["QoS"]["Phi_min_int"] / 50
        ram_needed = nsr["QoS"]["Phi_min_ext"] / 50
        bw_needed = (nsr["QoS"]["Phi_min_int"] + nsr["QoS"]["Phi_min_ext"]) / 2

        state["remaining_cpu_cores"] -= cpu_needed
        state["remaining_ram_cores"] -= ram_needed
        state["remaining_cpu_rans"] -= cpu_needed
        state["remaining_ram_rans"] -= ram_needed
        state["remaining_bandwith_tans"] -= bw_needed
        return state

# Example usage
if __name__ == "__main__":
    ql = AdmissionControlQLearning()

    # Example initial state
    state = {
        "remaining_cpu_cores": 16,
        "remaining_ram_cores": 64,
        "remaining_cpu_rans": 16,
        "remaining_ram_rans": 64,
        "remaining_bandwith_tans": 1000,
        "NSR": {
            "id": 1,
            "QoS": {
                "L_max_int": random.uniform(1, 10),
                "L_max_ext": random.uniform(1, 10),
                "Phi_min_int": random.uniform(50, 200),
                "Phi_min_ext": random.uniform(50, 200),
                "P_max_int": random.uniform(0, 0.01),
                "P_max_ext": random.uniform(0, 0.01)
            },
            "T0": random.randint(5, 15) * 60
        }
    }

    for episode in range(1000):
        action = ql.choose_action(state)
        reward = ql.reward_function(state, action)
        next_state = state.copy()
        if action == "Accept":
            next_state = ql.apply_nsr(next_state, next_state["NSR"])
        ql.update_q(state, action, reward, next_state)
        state = next_state
