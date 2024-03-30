"""
This script contains a Deep Q Network class which inherits from torch.nn.Module.
It is based on the network Q-Network found on 
https://github.com/hungtuchen/pytorch-dqn/blob/master/dqn_model.py.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DQN(nn.Module):
    """
    Deep Q Network class
    ...

    Atributes
    ---------
        input_shape: Input image's shape
        nb_actions: number of action-value to output. Pong has 6 actions by default.

    Methods
    -------
    forward(x)
        Defines the forward poass of the neural network.

    """
    def __init__(self, input_shape: tuple, nb_actions: int, params: dict) -> None:
        """
        Initializes the Deep Q-learning Network.
        
        :Parameters:
            input_shape: Input image's size
            nb_actions: number of action-value to output. Pong has 6 actions by default.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(input_shape[1] * input_shape[2] * 64, params['hidden_size'])
        self.fc5 = nn.Linear(params['hidden_size'], nb_actions)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        The forward function defines the forward pass of the neural network, specifying 
        how input data is processed through the layers to produce output predictions.

        :Parameters:
            x: Input image as tensor.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))

        return self.fc5(x)
