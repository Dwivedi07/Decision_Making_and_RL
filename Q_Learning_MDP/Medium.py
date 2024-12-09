import sys
import csv
import numpy as np
import pandas as pd
import math
import random
import time
from matplotlib import pyplot as plt
from MDP_med import MDP

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
        Q_s_a = np.zeros((len(self.MDP.states), len(self.MDP.actions)), dtype=float)
        
        # Group history data by episode index
        history_d = self.MDP.History  # (s, a, r, sp, ep) data
        episodes = {}
        for entry in history_d:
            s, a, r, sp, ep = entry
            if ep not in episodes:
                episodes[ep] = []
            episodes[ep].append((s, a, r, sp))
        
        # Iterate through the number of passes
        for i in range(self.MDP.k_max):
            print(f'##############  Pass Count: {i+1}#############\n')
            
            
            for ep_index, episode_data in episodes.items():
                # print(f'--- Processing Episode {ep_index} ---')

                # Process each transition in the episode
                for idx, (s, a, r, sp) in enumerate(episode_data):
                    # Check if this is the terminal transition of the episode
                    is_terminal = idx == (len(episode_data) - 1)

                    ######## Q-Learning Update #######
                    if is_terminal:
                        # For terminal states, Q(s, a) = r
                        Q_s_a[s-1, a-1] = r
                    else:
                        # Standard Q-learning update for non-terminal states
                        Q_s_a[s-1, a-1] = Q_s_a[s-1, a-1] + (1/(idx+1)**(0.77)) * (
                            r + self.MDP.gamma * max([Q_s_a[sp-1, ap-1] for ap in self.MDP.actions]) - Q_s_a[s-1, a-1]
                        )

        return Q_s_a
    
    def Q_nv(self,Q_s_a):
        '''
        Locate the same postion avergae out Q_s_a value and take the maximum action accordingly
        '''
        for s in range(len(self.MDP.states)):
            if np.all(Q_s_a[s] == 0):
                Q_pos = []
                pos, vel = s%500, s//500
                
                iter = np.linspace(-20,20,41)
                for v in range(100):
                ################################
                # for i in iter:
                    # v = max(0, min(99, vel+i))
                ################################
                    s_vel = int(1 + pos + 500*(v))
                    Q_pos.append(Q_s_a[s_vel-1])

                # Q_s_a[s] = np.mean(np.array(Q_pos), axis=0)
                Q_s_a[s] = np.max(np.array(Q_pos), axis=0)  # This one gave better result

        return Q_s_a
        

    def return_policy(self, Q_s_a):
        policy = np.zeros(len(self.MDP.states))
        
        for i in range(len(self.MDP.states)):
            if np.all(Q_s_a[i] == 0):
                policy[i] = random.randint(1,7)  #random exploration for non-visited states
            else:
                policy[i] = np.argmax(Q_s_a[i]) + 1 
            # policy[i] = np.argmax(Q_s_a[i]) + 1 

        return policy



def main():
    if len(sys.argv) != 4:
        raise Exception("usage: python Medium.py <infile>.csv <outfile>.policy <small/medium/large>")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    type_string = sys.argv[3]
    
    P = MDP(inputfilename, type_string)
    RL_agent = Intelligent_Agent(P, outputfilename, type_string)

if __name__ == '__main__':
    main()
