import sys
import csv
import numpy as np
import pandas as pd
import math
import os
import random
import time
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from MDPQN import MDP
from model import DQNet

class Intelligent_Agent():
    def __init__(self, MDP, model, criterion, optimizer, state_shape):
        self.MDP = MDP
        self.model = model
        self.criterion = criterion  # Loss function
        self.optimizer = optimizer  # Optimizer
        self.shape = state_shape
        
        start_time = time.time()
        self.DQN_train()
        end_time = time.time()
        time_taken = end_time - start_time
        print('This is the total run time:', time_taken)



    def DQN_train(self):
        print('################ Training Starts #####################')
        for e in range(self.MDP.k_max):
            print(f'################ Episode - {e+1} ##############\n')
            np.random.shuffle(self.MDP.History)
            batch_sample = self.MDP.History[0:self.MDP.batchsample]
            
            for his in batch_sample:
                state = torch.tensor(his[0], dtype=torch.float32).unsqueeze(0)  # Current state as tensor
                action = his[1]-1  # Action taken
                reward = his[2]  # Reward received
                next_state = torch.tensor(his[3], dtype=torch.float32).unsqueeze(0)  # Next state as tensor

                state = state.to(self.model.device)
                next_state = next_state.to(self.model.device)
                # Forward pass: Predict Q-values for the current state


                if self.shape == None:
                    q_values = self.model(state)
                     
                    # Compute the Q-target using Bellman optimality equation
                    with torch.no_grad():
                        next_q_values = self.model(next_state)

                        max_next_q_value = next_q_values.max().item()
                        q_target = reward + self.MDP.gamma * max_next_q_value


                    q_target_tensor = torch.tensor(q_target, dtype=torch.float32).to(self.model.device)  # Convert to tensor
                    # Calculate the loss: Only the Q-value of the taken action is updated
                    q_values[action] = q_target_tensor # Update Q-value for the action taken
                    
                    # Zero out gradients, perform backpropagation, and update weights
                    self.optimizer.zero_grad()
                    loss = self.criterion(q_values, q_values.detach())
                    loss.backward()
                    self.optimizer.step()
                
                else:
                    ################################################################  
                    ######### For breaking states into features ########
                    ################################################################
                    flat_index = int(state.item())-1
                    multi_indices = np.unravel_index(flat_index, self.shape)
                    state_idx_tensor = torch.tensor(multi_indices, dtype=torch.float32) 

                    flat_index2 = int(next_state.item())-1 
                    multi_indices2 = np.unravel_index(flat_index2, self.shape)
                    nextstate_idx_tensor = torch.tensor(multi_indices2, dtype=torch.float32) 
                    
                    #To cuda
                    state_idx_tensor = state_idx_tensor.to(self.model.device)
                    nextstate_idx_tensor = nextstate_idx_tensor.to(self.model.device)
                    ################################################################
                    ################################################################

                    q_values = self.model(state_idx_tensor) 
                    
                    # Compute the Q-target using Bellman optimality equation
                    with torch.no_grad():
                    
                        next_q_values = self.model(nextstate_idx_tensor)

                        max_next_q_value = next_q_values.max().item()
                        q_target = reward + self.MDP.gamma * max_next_q_value


                    q_target_tensor = torch.tensor(q_target, dtype=torch.float32).to(self.model.device)  # Convert to tensor
                    # Calculate the loss: Only the Q-value of the taken action is updated
                    q_values[action] = q_target_tensor # Update Q-value for the action taken
                    
                    # Zero out gradients, perform backpropagation, and update weights
                    self.optimizer.zero_grad()
                    loss = self.criterion(q_values, q_values.detach())
                    loss.backward()
                    self.optimizer.step()

    
    def return_policy(self):
        
        policy = np.zeros(len(self.MDP.states), dtype=int)
        
        for state_index in range(len(self.MDP.states)):
            if self.shape == None:
                state_tensor = torch.tensor(self.MDP.states[state_index], dtype=torch.float32).unsqueeze(0).to(self.model.device)

                with torch.no_grad(): 
                    q_values = self.model(state_tensor)

                # Find the action with the maximum Q-value
                best_action = torch.argmax(q_values).item()
                policy[state_index] = best_action 
 
            else:      
                # Convert the state_index to indices and then to a tensor and send to device
                multi_indices = np.unravel_index(state_index, self.shape)
                state_tensor = torch.tensor(multi_indices, dtype=torch.float32).unsqueeze(0).to(self.model.device)

            
                with torch.no_grad(): 
                    q_values = self.model(state_tensor)

                # Find the action with the maximum Q-value
                best_action = torch.argmax(q_values).item()
                policy[state_index] = best_action +1 

        return policy      
    
def writefile(policyfile, policy):
    with open(policyfile, 'w') as f:
        for action in policy:
            f.write("{}\n".format(int(action)))

def main():
    if len(sys.argv) != 4:
        raise Exception("usage: python RL.py <infile>.csv <outfile>.policy <small/medium/large>")
    
    if sys.argv[3] == 'small':
        state_shape = (10,10)
    elif sys.argv[3] == 'medium':
        state_shape = (500,100)
    else:
        state_shape = None


    # Check if GPU is being detected
    model_save_path = 'dqn_model_small.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('#############################################')
    print("device:", device)
    print('#############################################')

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    type_string = sys.argv[3]
    
    MDP_ = MDP(inputfilename, type_string)
    NN_model = DQNet(MDP_,device).to(device)

    # Load the saved model parameters if they exist
    # if os.path.isfile(model_save_path):
    #     NN_model.load_state_dict(torch.load(model_save_path, map_location=device))
    #     NN_model.eval()  
    #     print(f'Model parameters loaded from {model_save_path}')
    # else:
    #     print(f'No saved model found at {model_save_path}, initializing a new model.')
    
    print(NN_model)
    # for name, param in NN_model.named_parameters():
    #         if param.requires_grad:  # Only print weights that are trainable
    #             print(f"{name}: {param.data}")

    # Instantiate Criterion and Optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = torch.optim.Adam(NN_model.parameters(), lr=0.001)  # Optimizer with learning rate

    RL_agent = Intelligent_Agent(MDP_, NN_model, criterion, optimizer, state_shape)

    # Save the model parameters after the agent has trained 
    torch.save(NN_model.state_dict(), model_save_path)
    print(f'Model parameters saved to {model_save_path}')
    ######################################################
    writefile(outputfilename, RL_agent.return_policy())

if __name__ == '__main__':
    main()