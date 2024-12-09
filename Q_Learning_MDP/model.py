import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNet(nn.Module):
    '''
    input_dim=state_feature_size, where state_size is the size of the state representation
    Hidden layers - learns abstract features of the state that are useful for predicting Q-values
    Output_dim = action_size, one Q-value per action, corresponding to the expected reward for each possible action in the given state
    Ex: input_size=2, hidden_size=64, output_size=2
    '''
    def __init__(self, MDP_, device):
        super(DQNet, self).__init__()
        self.device = device
        input_size = 2
        output_size = len(MDP_.actions)
        hidden_neuron1 = 128
        hidden_neuron2 = 128
        hidden_neuron3 = 128
        hidden_neuron4 = 64

        self.fc1 = nn.Linear(input_size, hidden_neuron1)
        self.fc2 = nn.Linear(hidden_neuron1, hidden_neuron2)
        self.fc3 = nn.Linear(hidden_neuron2, hidden_neuron3)
        self.fc4 = nn.Linear(hidden_neuron3, hidden_neuron4)
        self.fc5 = nn.Linear(hidden_neuron4, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    


if __name__ == '__main__':
    
    model = DQNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)
