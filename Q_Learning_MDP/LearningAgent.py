import sys
import csv
import numpy as np
import pandas as pd
import math
import random
import time
from matplotlib import pyplot as plt
from MDP import MDP

class Intelligent_Agent():
    def __init__(self, MDP, fileout, type_string):
        self.policyfile = fileout
        self.MDP = MDP
        self.type = type_string
        ###################################
        start_time = time.time()
        ###################################
        Q_s_a = self.Q_learning()
        Q_s_a = self.Q_nv(Q_s_a)
        ###################################
        end_time = time.time()
        time_taken = end_time - start_time
        print('This is the total run time:', time_taken)
        ###################################

        self.policy = self.return_policy(Q_s_a)
        self.writefile()
    
    def writefile(self):
        with open(self.policyfile, 'w') as f:
            for action in self.policy:
                f.write("{}\n".format(int(action)))

    
    def Q_learning(self):
        Q_s_a = np.ones((len(self.MDP.states), len(self.MDP.actions)), dtype=float)
        Q_s_a = np.load(f'Q_valueslarge5175.npy')
        print(Q_s_a)

        for i in range(self.MDP.k_max):
            print(f'############## Episode {i+1}#############\n')
            history_d = self.MDP.History
            indices = np.random.choice(len(self.MDP.History), size=int(0.7 * len(self.MDP.History)), replace=False)
            history_d = self.MDP.History[indices]
    
            for idx, his in enumerate(history_d):
                s, a, r, sp = his[0], his[1], his[2], his[3]

                ######## Q- Learning Update#######              
                Q_s_a[s-1,a-1] = Q_s_a[s-1,a-1] + (self.MDP.alpha)*( r + self.MDP.gamma* max([Q_s_a[sp-1, ap-1] for ap in self.MDP.actions]) - Q_s_a[s-1,a-1])
        
        #Save the values after each iteration
        Q_values = np.array(Q_s_a)
        np.save(f'Q_valuesaverage{self.type}.npy', Q_values)
        return Q_s_a

    # def Sarsa(self):
    #     Q_s_a = np.zeros((len(self.MDP.states), len(self.MDP.actions)), dtype=float)
    #     Q_s_a = np.load(f'Q_values{self.type}.npy')
    #     # Group history data by episode index
    #     history_d = self.MDP.History  # (s, a, r, sp, ep) data
    #     episodes = {}
    #     for entry in history_d:
    #         s, a, r, sp, ep = entry  # Assumes `sp` and `ep` are included in `history_d`
    #         if ep not in episodes:
    #             episodes[ep] = []
    #         episodes[ep].append((s, a, r, sp))

    #     # Iterate through the number of passes
    #     for i in range(self.MDP.k_max):
    #         print(f'##############  Pass Count: {i+1}#############\n')

    #         for ep_index, episode_data in episodes.items():
                
    #             # Process each transition in the episode
    #             for idx, (s, a, r, sp) in enumerate(episode_data):
    #                 if idx < len(episode_data)-1:
    #                     ap =  episode_data[idx+1][1]
    #                     ######## SARSA- Learning Update#######              
    #                     Q_s_a[s-1,a-1] = Q_s_a[s-1,a-1] + (self.MDP.alpha)*( r + self.MDP.gamma* Q_s_a[sp-1, ap-1] - Q_s_a[s-1,a-1])
        
    #     Q_values = np.array(Q_s_a)
    #     np.save(f'Q_values{self.type}.npy', Q_values)
    #     return Q_s_a
    def Q_nv(self,Q_s_a):
            '''
            Locate the state avergae out Q_s_a value and take the maximum action accordingly
            '''
            for s in range(len(self.MDP.states)):
                if np.all(Q_s_a[s] == 1):
                    Q_pos = []
                    iter = np.linspace(-5000,5000,10001)
                    
                    ################################
                    for i in iter:
                        v = max(0, min(302019, int(i+s)))
                    ################################
                        s_vel = v
                        Q_pos.append(Q_s_a[v])

                    Q_s_a[s] = np.mean(np.array(Q_pos), axis=0)
                    # Q_s_a[s] = np.max(np.array(Q_pos), axis=0)  # This one gave better result

            return Q_s_a
    
    def return_policy(self, Q_s_a):
        policy = np.zeros(len(self.MDP.states))
        for i in range(len(self.MDP.states)):
            policy[i] = np.argmax([Q_s_a[i]]) + 1
        
        return policy



def main():
    if len(sys.argv) != 4:
        raise Exception("usage: python LearningAgent.py <infile>.csv <outfile>.policy <small/medium/large>")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    type_string = sys.argv[3]
    
    P = MDP(inputfilename, type_string)
    RL_agent = Intelligent_Agent(P, outputfilename, type_string)

if __name__ == '__main__':
    main()
