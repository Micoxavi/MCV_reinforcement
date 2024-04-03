"""
This script contains a memory buffer for the DQN which stores the states, actions, rewards
done conditions and next samples and is used on the training loop.
Once the buffer gets full, the first element of the queue is replaced.

The code for this memory buffer is based on 
https://github.com/PacktPublishing/Hands-on-Reinforcement-Learning-with-PyTorch

"""
import numpy as np


class ReplyBuffer:
    """
    Memory buffer class.
    ...

    Atributes
    ---------
        buffer_size: integer with the maximum memory the buffer can handle.
        

    Methods
    -------
        add_data(data)
            Function to add data to the memory_buffer.

        sample(batch_size)
            Number of data samples to return.
    """

    def __init__(self, buffer_size: int) -> None:
        """
        Buffer initializer function

        :Parameters:
            buffer_size: integer with the maximum memory the buffer can handle.

        """
        self.buffer = []
        self.max_size = buffer_size
        self.pointer = 0

    def add_data(self, data: tuple) -> None:
        """
        Function to add data to the memory_buffer.
        The data lists are appened to the buffer list when it is yet to reach 
        the maximum size. When it has reached it, the pointer will indicate which
        data position (oldest data) must be replaced.

        :Parameters:
            data: list with data to store.

        """

        if len(self.buffer) == self.max_size:
            self.buffer[int(self.pointer)] = data
            self.pointer = (self.pointer + 1) % self.max_size

        else:
            self.buffer.append(data)

    def sample(self, batch_size: int) -> tuple:
        """
        Function to obtain a random data sample from the memory buffer

        :Parameters:
            batch_size: Number of data samples to return.

        :Return:
            tuple with np.ndarray for each data element
        """
        index = np.random.randint(0, len(self.buffer) -1, size=batch_size)
        state_list, next_list, action_list, reward_list, done_list = [], [], [], [], []

        for i in index:
            state, next_state, action, reward, done = self.buffer[i]
            # TODO: Why does the tensor convert into a tuple at some point?

            if isinstance(state, tuple):
                state = state[0]
        
            state_list.append(np.array(state, copy=False))  # lista con arrays
            
            # state_array = np.append(state_array, state)
            next_list.append(np.array(next_state, copy=False))
            action_list.append(np.array(action, copy=False))
            reward_list.append(np.array(reward, copy=False))
            done_list.append(np.array(done, copy=False))


        # turn the lists into array
        # state_array = [np.array(arr) for arr in state_list]
        state_array = np.array(state_list)
        next_array = np.array(next_list)
        action_array = np.array(action_list).reshape(-1,1)
        reward_array = np.array(reward_list).reshape(-1,1)
        done_array = np.array(done_list).reshape(-1,1)

        return state_array, next_array, action_array, reward_array, done_array
