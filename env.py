# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 12:12:15 2018

@author: gurunath.lv
"""
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def cycle(iterable):
    while True:
        for x in iterable:
            yield x



def f2_score(y_true, y_pred, threshold=0.5):
    return fbeta_score(y_true, y_pred, 2, threshold)


def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=0)
    precision = true_positive.div(y_pred.sum(dim=0).add(eps))
    recall = true_positive.div(y_true.sum(dim=0).add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))


def tptn(y_true, y_pred, threshold=0.5, eps=1e-9):
#    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()
#    print(y_pred,y_true)    
    true_positive = (y_pred * y_true).sum()
    
#    precision = true_positive.div(y_pred.sum(dim=0).add(eps))
#    recall = true_positive.div(y_true.sum(dim=0).add(eps))
    true_negative=((y_pred==0)*(y_true==0)).float().mean()
#    print('neg:',true_negative,true_negative*0.2,'Pos:',true_positive)
#    print(true_positive,true_negative)
    return true_negative*0.2
def cross_entropy(y_true,logits):
    #weight=torch.Tensor([0.3,0.7])
    return (1-F.cross_entropy(logits,y_true,))
#output = torch.randn(10, 120).float()
#target = torch.FloatTensor(10).uniform_(0, 120).long()

#loss = criterion(output, target)

class AnomalyEnv(gym.Env):
    def __init__(self,dataloader,writer):
        self.dataloader=dataloader
#        self.action_space=[0,1]
        self.global_rewards=[]
        self.writer=writer
        self.i=0
    def step(self,action):
#        assert action in self.action_space
#        if self.i==0:
#            state,_=next(cycle(self.dataloader))
#            return state,0
#        print(action,self.true_action.shape)
#        rewards=[]
#        print(action.size(),true_action.size())
#        mask=action.squeeze()==self.true_action.squeeze()
        reward=cross_entropy(self.true_action.long().squeeze(),action)
#        true_action+=0.02
#        mask=torch.LongTensor(mask.long())
#        mask=mask*true_action
#        print(mask.shape,true_action.shape)
#        reward=self.true_action[mask].squeeze().float().sum()
#        reward=mask.float().mean()
        state,self.true_action=next(cycle(self.dataloader))
#        reward=torch.mean(mask.float())
#        self.i
#        for ai,yi in zip(action,true_action):
#            if not isinstance(ai,int):
#                ai,yi=ai.squeeze(),yi.squeeze()
#                print(ai,yi)
#            if ai==yi==1:
#                rew=1
##                rewards.append(1)
#            if ai==yi==0:
#                rew=0
##                rewards.append(0.1)
#            if ai!=yi:
#                rew=-1
        self.i+=1
#            rewards.append(rew)
        self.writer.add_scalar('rewards',reward,self.i)
        self.global_rewards.append(reward)
#        print(self.i,reward)
        return state,reward
    def render(self,close=True):
        print("REwards so far {}".format(np.mean(self.global_rewards)))
        self.writer.add_histogram('global_rewards',np.array(self.global_rewards))
        
#        plt.plot(self.global_rewards)
#        plt.title('Rewards so far!!!')
#        plt.show()
    def reset(self):
        self.dataloader=iter(cycle(self.dataloader))
        state,self.true_action=next(self.dataloader)
        return state


#class ForecastEnv(gym.Env):
#    def __init__(self,dataloader,writer):
#        self.writer=writer
#        self.dataloader=dataloader
#        self.global_rewards=[]
#        self.i=0
#    def step(self,action):
#        state,true_action=next(self.dataloader)
#        
#        
#    def render(self,close=True):
#        print("REwards so far {}".format(np.mean(self.global_rewards)))
#        self.writer.add_histogram('global_rewards',np.array(self.global_rewards))
#        
##        plt.plot(self.global_rewards)
##        plt.title('Rewards so far!!!')
##        plt.show()
#    def reset(self):
#        self.dataloader=iter(cycle(self.dataloader))
#        state,_=next(self.dataloader)
#        return state
#    
#
#
#
#
#
#
#




class GuessingGame(gym.Env):
    """Number guessing game
    The object of the game is to guess within 1% of the randomly chosen number
    within 200 time steps
    After each step the agent is provided with one of four possible observations
    which indicate where the guess is in relation to the randomly chosen number
    0 - No guess yet submitted (only after reset)
    1 - Guess is lower than the target
    2 - Guess is equal to the target
    3 - Guess is higher than the target
    The rewards are:
    0 if the agent's guess is outside of 1% of the target
    1 if the agent's guess is inside 1% of the target
    The episode terminates after the agent guesses within 1% of the target or
    200 steps have been taken
    The agent will need to use a memory of previously submitted actions and observations
    in order to efficiently explore the available actions
    The purpose is to have agents optimise their exploration parameters (e.g. how far to
    explore from previous actions) based on previous experience. Because the goal changes
    each episode a state-value or action-value function isn't able to provide any additional
    benefit apart from being able to tell whether to increase or decrease the next guess.
    The perfect agent would likely learn the bounds of the action space (without referring
    to them explicitly) and then follow binary tree style exploration towards to goal number
    """
    def __init__(self):
        self.range = 1000  # Randomly selected number is within +/- this value
        self.bounds = 10000

        self.action_space = spaces.Box(low=np.array([-self.bounds]), high=np.array([self.bounds]))
        self.observation_space = spaces.Discrete(4)

        self.number = 0
        self.guess_count = 0
        self.guess_max = 200
        self.observation = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        if action < self.number:
            self.observation = 1

        elif action == self.number:
            self.observation = 2

        elif action > self.number:
            self.observation = 3

        reward = 0
        done = False

        if (self.number - self.range * 0.01) < action < (self.number + self.range * 0.01):
            reward = 1
            done = True

        self.guess_count += 1
        if self.guess_count >= self.guess_max:
            done = True

        return self.observation, reward, done, {"number": self.number, "guesses": self.guess_count}

    def reset(self):
        self.number = self.np_random.uniform(-self.range, self.range)
        self.guess_count = 0
        self.observation = 0
        return self.observation

if __name__=='__main__':
    gg=GuessingGame()