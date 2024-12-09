import random
import numpy as np
import pandas as pd
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

class DQNAgent:
    def __init__(self, state_size, action_size, memory):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = memory  # Use preloaded memory
        self.gamma = 0.95    # discount rate
        self.epsilon = 0     # No exploration as we're using offline data
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            state_break = np.array(np.unravel_index(state, (10,10))).reshape(1,-1)  #Smalldata
            next_state_break = np.array(np.unravel_index(next_state, (10,10))).reshape(1,-1)  #Smalldata
            target = (reward + self.gamma * np.amax(self.model.predict(next_state_break)[0]))
            target_f = self.model.predict(state_break)
            target_f[0][action] = target
            self.model.fit(state_break, target_f, epochs=1, verbose=0)


# Load and preprocess data from CSV
def load_data(csv_path, state_size):
    data = pd.read_csv(csv_path)
    memory = deque(maxlen=40000)
    
    for index, row in data.iterrows():
        s = int(row['s'])
        a = int(row['a'])
        r = float(row['r'])
        sp = int(row['sp'])
        # done = False  # Assuming non-terminal states; update if your data includes terminal info
        memory.append((s-1, a-1, r, sp-1))
    
    return memory

def writefile(policyfile, policy):
    with open(policyfile, 'w') as f:
        for action in policy:
            f.write("{}\n".format(int(action)))


if __name__ == "__main__":
    state_size = 1  # Adjust to match state dimension in your data
    state_size_NN = 2  # Adjust to match state dimension in your data
    action_size = 4  # Adjust to match action dimension in your data
    csv_path = "data/small.csv"

    # Load experiences from CSV and initialize memory
    memory = load_data(csv_path, state_size)

     # Load or create a new model
    try:
        agent = DQNAgent(state_size_NN, action_size, memory)
        agent.model = load_model("dqn_model.h5")  # Load the model
        print("Loaded existing model from dqn_model.h5")
    except Exception as e:
        print("Could not load model. Creating a new one.")
        agent = DQNAgent(state_size_NN, action_size, memory)

    batch_size = 256
    num_epochs = 100  # Number of training iterations

    for epoch in range(num_epochs):
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        print(f"Epoch: {epoch + 1}/{num_epochs} complete")

    # Save the model after training
    agent.model.save("dqn_model.h5")
    print("Model saved as dqn_model_small_chatgpt.h5")

    # Compute the best action for each state in the dataset
    agent.model = load_model("dqn_model.h5")
    policy = []
    for s in range(100):
        s_break = np.array(np.unravel_index(s, (10,10))).reshape(1,-1)
        q_values = agent.model.predict(s_break)
        best_action = np.argmax(q_values[0])+1  # Best action for this state
        policy.append(best_action)

    # Write policy to a file
    writefile("chatgpt.policy", policy)
    print(f"Policy written")
