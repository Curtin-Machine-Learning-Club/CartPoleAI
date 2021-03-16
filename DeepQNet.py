"""
    Author    : Milan Marocchi
    Date      : 15/03/2021
    Purpose   : Contains the Deep Q Net classes
    Reference : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class DeepQNet(nn.Module):
    """
    class   : DeepQNet
    Purpose : Class for the deep q nueral network
    """

    def __init__(self, inputSize, hiddenSize, outputSize):
        """
        Creates an instance of the deep q net
        :param inputSize: Size of the input layer
        :param hiddenSize: Size of the hidden layer
        :param outputSize: Size of the output layer
        """
        super(DeepQNet, self).__init__()

        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        """
        Returns the output tensors from input tensors
        :return:
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, filename='model.pth'):
        """
        Saves the model onto the computer
        : param filename: The filename of the model to be saved
        """
        modelFolderPath = "./model"

        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)

        filename = os.path.join(modelFolderPath, filename)
        torch.save(self.state_dict(), filename)
