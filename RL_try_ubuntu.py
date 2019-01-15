# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:53:39 2018

@author: gurunath.lv
"""

import argparse
#import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.datasets import make_classification
from env import AnomalyEnv
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from torch.autograd import Variable
#from imblearn.datasets import make_imbalance
from tensorboardX import SummaryWriter
import pandas as pd

writer=SummaryWriter('RL-try')


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

#env=GuessingGame()
#env = gym.make('CartPole-v0')
#env.seed(args.seed)
#torch.manual_seed(args.seed)
#
class AnomDataset(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.Y=y

    def __getitem__(self,index):
        return Variable(torch.FloatTensor(self.X[index])),Variable(torch.LongTensor([self.Y[index]]))
    def __len__(self):
        return self.X.shape[0]
    
    
#dummy_data=make_classification(n_classes=2,n_samples=10000,n_features=5,)
#state,reward=aenv.step(y)
#aenv.render()

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(5, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)



def select_action(state):
#    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
#    print(list(policy.named_parameters()))
#    print(probs)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action


def finish_episode(episode):
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:#Reverse the list
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
#    i=0
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        loss=-log_prob * reward
        policy_loss.append(loss)
        writer.add_histogram('trainloss',loss.detach().numpy())
#        i+=1
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in tqdm(range(20),desc='Epochs'):
        state = env.reset()
        for t in tqdm(range(int(X.shape[0]/1000)),desc='iterations'):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
#            if done:
#                break
            if (t+1)%100==0:
                print("Back-propagating-Errors")
                finish_episode(i_episode)
        running_reward = running_reward * 0.99 + t * 0.01
        
        for idx,p in enumerate(policy.parameters()):
            writer.add_histogram(f'model_parameters{idx}',p.grad.detach().numpy())
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
#        if running_reward > env.spec.reward_threshold:
#            print("Solved! Running reward is now {} and "
#                  "the last episode runs to {} time steps!".format(running_reward, t))
#            break


if __name__ == '__main__':
    data=pd.read_feather(r'../tagged_data_feather')
    cont_names=['BerPreFecMax','PhaseCorrectionAve','PmdMin','Qmin','SoPmdAve']

    X,y=data[cont_names].fillna(0),data['tagged_alerts']
#    print(X.isnull().sum(),y.isnull().sum())
    X,y=X.values,y.values
    
#    print(X.dtype)
#    X,y=make_imbalance(X,y,{0:4000,1:500})
    anomdataset=AnomDataset(X,y)
    data_loader=DataLoader(anomdataset,batch_size=1000)    
    env=AnomalyEnv(data_loader,writer)
    policy = Policy()
    policy=policy.apply(init_weights)
#    print('check',list(policy.named_parameters()))
    
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    eps = np.finfo(np.float32).eps.item()


    main()
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     