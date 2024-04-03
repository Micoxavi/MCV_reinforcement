"""
This script contains a DQNAgent class
"""

import torch
from torch import optim
from torch.nn import functional as F
import numpy as np

from model import DQN


class DQNAgent:
    """
    The DQNAgent agent class. Implements the Q-learning algorithm 
    for reinforcement learning tasks.
    ...

    Atributes
    ---------
        input_shape: Input image's shape.
        nb_actions: number of action-value to output. Pong has 6 actions by default.
        device: cuda when gpu available, cpu otherwise.
        params: Dictionary with the different hyperparameters configuration.

    Methods
    -------
    select_actions(state)
        Selects actions based on the epsilon-greedy method and the current state.

    train(replay_buffer, batch_size, discount)
        Agent training function.

    update_target_network(nb_iterations, update_every)
        Funtion to update the self.target_network fully with the train newtork params.
        
    """

    def __init__(self, input_shape: int, nb_actions: int, device: str, params: dict, env) -> None:
        """
        DQNAgent initialization

        :Parameters:
            input_shape: Input image's shape.
            nb_actions: number of action-value to output. Pong has 6 actions by default.
            device: cuda when gpu available, cpu otherwise.
            params: Dictionary with the different hyperparameters configuration.

        """
        self.device = device
        self.params = params
        self.env = env

        # @train_network -> Used for action selection during the agent's interaction with
        # the enviroment. Its parameters are updated based on the TD-error calculated during
        # training.
        self.train_network = DQN(input_shape, nb_actions, params).to(self.device)

        # @target_network -> used for estimating the target Q-values during the training process
        # its parameters are are periodically updated by copying the parameters from the
        # train_network, helping to stabilize the training process.
        self.target_network = DQN(input_shape, nb_actions, params).to(self.device)
        self.target_network.load_state_dict(self.train_network.state_dict())

        self.optimizer = optim.Adam(self.train_network.parameters(), lr=self.params['learning_rate'])

    def select_actions(self, state: any) -> int:
        """
        Function used to select actions based on the epsilon-greedy method.
        When the epsilon is bigger than a random number, the action is taken randomly
        and the agent only observes. When epsilon is smaller, the agent decides the action.

        :Parameters:
            state: Actual bar position on the image.

        """
        if np.random.rand() <= self.params['epsilon_start']:
            action = self.env.action_space.sample()

        else:
            with torch.no_grad():
                input_state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)

                # pylint: disable=not-callable
                action = int(self.train_network(input_state).forward().max(1)[1])
                # pylint: enable=not-callable

        return action

    def train(self, replay_buffer: object, batch_size: int, discount) -> float:
        """
        Agent training function. As every agent can be reused for different tasks, the training
        function is usually encapsulated within the agent itself for easier modularity.

        :Parameters:
            replay_buffer: Memory buffer with tuples (state, next_state, action, reward, done)
            batch_size: size of the data sampling batch
            discount: reward reduction parameter.         
        """
        state, next_state, action, reward, done = replay_buffer.sample(batch_size)

        state_batch = torch.FloatTensor(state).to(self.device)
        next_state_batch = torch.FloatTensor(next_state).to(self.device)

        # the output tensors have the dimensions permuted so we have to fix it.
        reshaped_state_batch = state_batch.permute(0, 3, 1, 2)
        reshaped_next_state_batch = next_state_batch.permute(0, 3, 1, 2)

        action_batch = torch.LongTensor(action).to(self.device)
        reward_batch = torch.FloatTensor(reward).to(self.device)
        done_batch = torch.FloatTensor(1. - done).to(self.device)

        # Get train Q network values for the current state and action.

        # pylint: disable=not-callable
        train_values = self.train_network(reshaped_state_batch).gather(1, action_batch)

        # Get the target Q network values.
        # @(reward_batch + done_batch * discount) --> immediate reward + discount
        # @(torch.max(self.target_network(next_state_batch).detach()) --> next_states Q values

        with torch.no_grad():

            target_values = reward_batch + done_batch * discount * \
                             torch.max(self.target_network(reshaped_next_state_batch).detach(),
                                       dim=1)[0].view(batch_size, -1)
        # pylint: enable=not-callable

        # Get the loss between train Q network and target Q network. DQN typically uses either
        # Huber loss, or MSE loss. @smooth_l1_loss --> similar to Huber.

        loss = F.smooth_l1_loss(train_values, target_values)

        # Optimize the parameters with the loss

        # Zero out the gradient to prevent errors from accumulation to previous values.
        self.optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping to avoid exploding gradients.
        # @clamp_ --> Operation done in-place modifying original gradient data.
        for param in self.train_network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        # return the loss to monitor it.
        return loss.detach().cpu().numpy()

    def update_target_network(self, nb_iterations: int, update_every: int):
        """
        Funtion to update the self.target_network fully with the train newtork params.

        :Parameters:
            nb_iterations:
            update_every: number of optimization steps for the target network to be updated
        """
        if nb_iterations % update_every == 0:
            # print("Updating target network parameters")
            self.target_network.load_state_dict(self.train_network.state_dict())
