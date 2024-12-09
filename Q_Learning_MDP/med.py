import numpy as np
import pandas as pd

class MountainCarAgent:
    def __init__(self, n_states=50000, n_actions=7):
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions))
        
    def train(self, data_file, learning_rate=0.01, episodes=300):
        """
        Train the agent using the provided CSV data
        """
        # Read the CSV file
        data = pd.read_csv(data_file)
        
        # Convert data to numpy arrays for faster processing
        states = data['s'].values
        actions = data['a'].values
        rewards = data['r'].values
        next_states = data['sp'].values
        
        # Training loop
        for i in range(episodes):
            print(i)
            
            for state, action, reward, next_state in zip(states, actions, rewards, next_states):
                # Q-learning update
                best_next_action = np.argmax(self.q_table[next_state])-1
                current_q = self.q_table[state, action-1]
                next_max_q = self.q_table[next_state, best_next_action]
                
                # Update Q-value
                new_q = current_q + learning_rate * (reward + next_max_q - current_q)
                self.q_table[state, action-1] = new_q
    
    def get_policy(self):
        """
        Return the deterministic policy based on the learned Q-values
        """
        return np.argmax(self.q_table, axis=1)
    
    def get_action(self, state):
        """
        Get action for a given state using the learned policy
        """
        return np.argmax(self.q_table[state])

# def create_policy_function(csv_file):
#     """
#     Train the agent and return a policy function that can be used with the environment
#     """
#     # Initialize and train the agent
#     agent = MountainCarAgent()
#     agent.train(csv_file)
    
#     def policy_function(observation):
#         # Convert continuous observation to discrete state
#         pos = int((observation[0] + 1.2) * 250)  # Assuming position range [-1.2, 0.6]
#         vel = int((observation[1] + 0.07) * 714)  # Assuming velocity range [-0.07, 0.07]
        
#         # Ensure values are within bounds
#         pos = max(0, min(499, pos))
#         vel = max(0, min(99, vel))
        
#         # Calculate state integer
#         state_int = 1 + pos + 500 * vel
        
#         # Get action from policy
#         return agent.get_action(state_int)
    
#     return policy_function

def writefile(policyfile, policy):
    with open(policyfile, 'w') as f:
        for action in policy:
            f.write("{}\n".format(int(action)))


if __name__ == "__main__":
    # Create and train the agent
    agent = MountainCarAgent()
    agent.train("data/medium.csv")
    
    # Get the deterministic policy
    policy = agent.get_policy()
    print(policy)

    writefile("dat.policy", policy)

    # # Create a policy function that can be used with the environment
    # policy_function = create_policy_function("data/medium.csv")
    
    # # Example of how to use the policy function
    # example_observation = np.array([-0.5, 0.0])
    # action = policy_function(example_observation)
    # print(f"Action for observation {example_observation}: {action}")
    
    # # You can also save the policy for later use
    # np.save("mountain_car_policy.npy", policy)