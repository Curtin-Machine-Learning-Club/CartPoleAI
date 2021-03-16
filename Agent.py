"""
    Author    : Milan Marocchi
    Date      : 15/03/2021
    Purpose   : Contains the Agent class for the cartpole enviroment
    Reference : https://github.com/Lazydok/RL-Pytorch-cartpole/blob/master/1_dqn.py
                https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import gym
from DeepQNet import *
from ReplayMemory import *
from QTrainer import *
import torch
import random
import math
import sys

class Agent:
    """
    class   : Agent
    purpose : This class represents the agent in the cartpole environment
    """

    def __init__(self):
        """
        Creates an instance of Agent and initialises it
        """

        self.games = 0
        self.completedSteps = 0

        self.epsilon = 0.001
        self.gamma = 0.85

        self.model = DeepQNet(4, 148, 2)
        self.trainer = QTrainer(self.model, self.epsilon, self.gamma)
        self.environment = gym.make('CartPole-v0').unwrapped
        self.memory = ReplayMemory(10000)

        self.ed = []

    def runEpisode(self, episode, train=True):
        """
        Runs one episode
        :param train: If the agent is training will be true otherwise false
        :return: Returns if the episode is finished or not
        """
        output = False
        state = self.environment.reset()
        steps = 0

        while True:
            self.environment.render()

            action = self._getAction(torch.FloatTensor([state]), train)
            nextState, reward, done, _ = self.environment.step(action[0, 0].tolist())

            # negative reward if lose
            if done:
                if steps < 30:
                    reward -= 10
                else:
                    reward -= 1
            if steps > 100:
                reward += 1
            if steps > 200:
                reward += 1
            if steps > 300:
                reward += 1

            self.memory.push((torch.FloatTensor([state]),
                              action,
                              torch.FloatTensor([nextState]),
                              torch.FloatTensor([reward])))

            self.trainMemory()

            state = nextState
            steps += 1

            if done or steps >= 1000:
                print("[Episode {:>5}] steps: {:>5}".format(episode, steps))
                if sum(self.ed[-10:]) /10 > 800:
                    return True
                break
        return False


    def _getAction(self, state, train=True):
        """
        Returns the action to be take
        :param state: The current state of the agen
        :param train: If the agent is training will be true
        :return: action to be taken
        """
        sample = random.random()
        threshold = 0.9 + (0.9 - 0.05) * math.exp(-1 * self.completedSteps / 200)
        self.completedSteps = self.completedSteps + 1

        if train:
            if sample > threshold:
                with torch.no_grad():
                    action = self.model(Variable(state).type(torch.FloatTensor)).data.max(1)[1].view(1, 1)
            else:
                action = torch.LongTensor([[random.randrange(2)]])
        else:
            with torch.no_grad():
                action = self.model(Variable(state).type(torch.FloatTensor)).data.max(1)[1].view(1, 1)

        return action


    def trainMemory(self):
        """
        Trains the memory using the trainer
        """

        self.trainer.trainStep(self.memory)
        self.model.save()

    def loadModel(self, filename='model.pth'):
        """
        Loads a model from a file and gets the agent to use it
        :param filename: The filename of the model file
        """
        modelFolderPath = './model'
        filename = os.path.join(modelFolderPath, filename)

        self.model.load_state_dict(torch.load(filename))
        self.model.eval()


def train(iterations):
    """
    Trains the model
    """
    agent = Agent()
    episode = 1

    for iteration in range(iterations):
        finished = agent.runEpisode(episode)
        episode += 1

        if finished:
            print("finished episode...")

def run():
    """
    Runs the model with out training it
    """
    agent = Agent()
    agent.loadModel()
    episode = 1

    while True:
        finished = agent.runEpisode(episode, train=False)
        episode += 1

        if finished:
            print("finished episode...")

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Incorrect arguments. Usage: python3 Agent.py run/train [iterations]")
    elif sys.argv[1] == 'run':
        run()
    elif sys.argv[1] == 'train':
        if len(sys.argv) < 3:
            print("Incorrect arguments. Usage: python3 Agent.py run/train [iterations]")
        if int(sys.argv[2]) > 0:
            train(int(sys.argv[2]))