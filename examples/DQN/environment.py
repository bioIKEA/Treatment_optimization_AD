from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random


class AlzheimersEnv(Env):
    def __init__(self):
        #discrete actions: 2 main medication, medication with sleep medication, no medication
        self.action_space=Discrete(4)
        #observation of MMSE score of patients 
        self.observation_space=Box(low=np.array([0]), high=np.array([30]))
        #initializing the MMSE score state
        self.state=26
        #lowest score will mean severe demetia
        self.lowest=8

    def step(self, action):
        self._take_action(action)
        if (self.state-self.original)<0:
            reward=-1
        elif (self.state-self.original)==0:
            reward=1
        else:
            reward=2
        if self.state<=self.lowest:
            done=True
        else:
            done=False

        info={}
        return self.state, reward, done, info

    def _take_action(self, action):
        self.original=self.state
        self.state+=action-1

    def render(self):
        pass
    def reset(self):
        self.state=26
        return self.state
