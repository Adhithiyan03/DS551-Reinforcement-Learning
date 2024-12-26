#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.conv=nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU(),
                )

        self.linear = nn.Sequential(
            nn.Linear(4*4*64, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def select_action(self,state,e):
        if random.random() > e:
            state_tensor = torch.FloatTensor(np.float32(state)).unsqueeze(0)
            q_value = self.forward(state_tensor)
            action = q_value.max(1)[1].item()
        else:
            action = random.randint(0, env.action_space.n - 1)
        
        return action

class Dueling_DQN(nn.Module):
    """Initialize a dueling deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(Dueling_DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.conv=nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU(),
                )

        self.advantage = nn.Sequential(
            nn.Linear(4*4*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions))

        self.value = nn.Sequential(
            nn.Linear(4*4*64, 512),
            nn.ReLU(),
            nn.Linear(512, 1))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        x = self.features(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()
    
    def select_action(self,state,e):
        if random.random() > e:
            state_tensor = torch.FloatTensor(np.float32(state)).unsqueeze(0)
            q_value = self.forward(state_tensor)
            action = q_value.max(1)[1].item()
        else:
            action = random.randint(0, env.action_space.n - 1)
        
        return action
