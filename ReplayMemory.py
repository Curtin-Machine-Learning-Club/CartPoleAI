"""
    Author    : Milan Marocchi
    Date      : 15/03/2021
    Purpose   : Contains the Deep Q Net classes
    Reference : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

from collections import deque
import random


class ReplayMemory:
    """
    class : Replay Memory
    purpose : Represents the memory of the model
    """
    def __init__(self, size):
        """
        Creates an instance of the replaymemory class
        :param size: The size of the memory
        """
        self.size = size
        self.memory = deque(maxlen=size)

    def push(self, transition):
        """
        Pushes a transition onto the memory
        :param transition: The transition to store in memory
        """
        # Deque automatically pops off furthest left element when size limit reached
        self.memory.append(transition)

    def sample(self, batchSize):
        """
        Returns a sample from the memory
        :param batchSize: The size of the memory to sample
        """
        return random.sample(self.memory, batchSize)

    def __len__(self):
        """
        Returns the length of the memory
        """
        return len(self.memory)
