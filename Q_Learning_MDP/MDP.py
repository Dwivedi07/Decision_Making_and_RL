import sys
import csv
import numpy as np
import pandas as pd
import math
import random



class MDP():
    def __init__(self, inputfile, ty_str):
        self.input_csv = inputfile
        self.datatype = ty_str
        self.state_data, self.action_data, self.reward_data, self.next_states_data= self.Extract_data()
        self.History = np.array([self.state_data, self.action_data, self.reward_data, self.next_states_data]).T
        self.alpha = 0.002  #decay considering
        self.k_max = 0  #1000 - small; 
        self.lambda_ = 0.7
        self.batchsample = int(len(self.History))
        
        self.features()

    def Extract_data(self):
        try:
            df = pd.read_csv(self.input_csv)
            states = df.iloc[:, 0].values
            actions = df.iloc[:, 1].values
            rewards = df.iloc[:, 2].values
            next_states = df.iloc[:, 3].values
            return states, actions, rewards, next_states

        except FileNotFoundError:
            print(f"File {self.inputfilepath} not found!")

    def features(self):
        if self.datatype == 'small':
            self.states = np.linspace(1,100,100,endpoint=True)
            self.actions = np.array([1,2,3,4])
            self.gamma = 0.95

        elif self.datatype == 'medium':
            self.states = np.linspace(1,50000,50000,endpoint=True)
            self.actions = np.array([1,2,3,4,5,6,7])
            self.gamma = 1

        else:
            self.states = np.linspace(1,302020 ,302020 ,endpoint=True)
            self.actions = np.array([1,2,3,4,5,6,7,8,9])
            self.gamma = 0.95