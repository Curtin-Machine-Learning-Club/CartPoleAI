"""
    Author    : Milan Marocchi
    Date      : 15/03/2021
    Purpose   : Contains the Deep Q Net classes
    Reference : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class QTrainer:
    """
    class   : QTrainer
    purpose : Trainer for the Q learning
    """

    def __init__(self, model, lr, gamma):
        """
        Creates an instance of the QTrainer
        :param model: The model being used
        :param lr: The learning rate value
        :param gamma: The gamma value
        """
        self.lr = lr
        self.gamma = gamma
        self.batchSize = 64
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

    def trainStep(self, memory):
        """
        Trains the model using one step
        :param memory: The memory of the agent
        """
        if len(memory) >= self.batchSize:
            transitions = memory.sample(self.batchSize)
            batchState, batchAction, batchNextState, batchReward = zip(*transitions)

            batchState = Variable(torch.cat(batchState))
            batchAction = Variable(torch.cat(batchAction))
            batchNextState = Variable(torch.cat(batchNextState))
            batchReward = Variable(torch.cat(batchReward))

            # Get the predicted Q values for the current state
            Qvalue = self.model(batchState).gather(1, batchAction).view(64)
            maxQValue = self.model(batchNextState).detach().max(1)[0]
            expectedQValue = batchReward + (self.gamma * maxQValue)

            loss = F.smooth_l1_loss(Qvalue, expectedQValue)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
