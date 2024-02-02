from collections import deque
import random


class ReplayBuffer:
    """
    A simple FIFO (first-in-first-out) replay buffer for storing experiences.

    Attributes:
        buffer_size (int): The maximum number of experiences the buffer can hold.
    """

    def __init__(self, buffer_size: int):
        """
        Initialize the replay buffer.

        Args:
            buffer_size (int): Maximum size of the buffer.
        """
        self.buffer = deque(maxlen=buffer_size)

    def iterate(self):
        """
        Iterate over the experiences in the buffer.

        Yields:
            tuple: Each experience in the buffer.
        """
        for experience in self.buffer:
            yield experience

    def add(self, experience):
        """
        Add a new experience to the buffer.

        Args:
            experience (tuple): A tuple representing an experience 
                                (state, action, reward, next_state, done).
        """
        self.buffer.append((experience))

    def sample(self, batch_size: int):
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            list: A list of sampled experiences.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """
        Get the current size of the replay buffer.

        Returns:
            int: The number of experiences currently in the buffer.
        """
        return len(self.buffer)
