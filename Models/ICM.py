import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

import numpy as np
target_path='./'
sys.path.append(target_path)

class Phi(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(Phi, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.normalize(x)
        y = F.sigmoid(self.fc1(x))
        y = F.sigmoid(self.fc2(y))
        y = F.sigmoid(self.fc3(y))
        y = F.sigmoid(self.fc4(y))
        return y


class Gnet(nn.Module):
    def __init__(self):
        super(Gnet, self).__init__()
        self.linear1 = nn.Linear(576, 256)
        self.linear2 = nn.Linear(256, 12)

    def forward(self, state1, state2):
        x = torch.cat((state1, state2), dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        y = F.softmax(y, dim=1)
        return y
    
class Fnet(nn.Module):
    def __init__(self):
        super(Fnet, self).__init__()
        self.linear1 = nn.Linear(300, 256)
        self.linear2 = nn.Linear(256, 288)
    
    def forward(self, state, action):
        action_ = torch.zeros(action.shape[0][2])
        indices = torch.stack((torch.arange(action.shape[0]), 
                                action.squeeze()), dim=0)
        indices = indices.tolist()
        action_[indices] = 1.
        x = torch.cat((state, action_), dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        return y


class ICMModule(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ICMModule, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(128 + action_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128)
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state, next_state, action):
        state_feature = self.feature(state)
        next_state_feature = self.feature(next_state)
        predicted_next_state_feature = self.forward_model(torch.cat([state_feature, action], dim=1))
        predicted_action = self.inverse_model(torch.cat([state_feature, next_state_feature], dim=1))
        forward_loss = nn.MSELoss()(predicted_next_state_feature, next_state_feature)
        inverse_loss = nn.CrossEntropyLoss()(predicted_action, action.argmax(dim=1))
        return forward_loss + inverse_loss, forward_loss