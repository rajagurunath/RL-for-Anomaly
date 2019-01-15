# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:53:39 2018

@author: gurunath.lv
"""

import argparse
#import gym
import numpy as np
from itertools import count
import sys
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
from imblearn.datasets import make_imbalance
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


#class Policy(nn.Module):
#    def __init__(self):
#        super(Policy, self).__init__()
#        self.affine1 = nn.Linear(1, 10)
#        self.affine2 = nn.Linear(10, 2)
#
#        self.saved_log_probs = []
#        self.rewards = []
#
#    def forward(self, x):
#        x = F.relu(self.affine1(x))
#        action_scores = self.affine2(x)
#        return F.softmax(action_scores, dim=1)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv=nn.Conv1d(1,10,1)
        self.affine1 = nn.Linear(100,2)
#        self.affine2 = nn.Linear(10, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x=F.relu(self.conv(x.view(10,1,-1))).view(1,-1)
        
        action_scores = self.affine1(x)
#        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
policy=policy.apply(init_weights)
optimizer = optim.Adam(policy.parameters(), lr=1e-3,amsgrad=True,weight_decay=0.02)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
#    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
#    m = Categorical(probs)
#    action = m.sample()
    policy.saved_log_probs.append(torch.max(probs))
    return probs


def finish_episode(episode):
    R = 0
    policy_loss = []
    rewards = []
#    print('saved',policy.saved_log_probs)
    for r in policy.rewards[::-1]:#Reverse the list
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    i=0
#    print('rew',rewards)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        loss= -log_prob * reward
#        print('loss',loss,loss.reshape(-1,1).shape)
#        print('loss',loss.shape)
        policy_loss.append(loss.reshape(-1,1))
        writer.add_scalar('trainloss',loss.mean().item(),episode+i)
        i+=1
    optimizer.zero_grad()
#    print(policy_loss)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
#    print(list(policy.named_parameters()))
    for i_episode in tqdm(range(10),desc='Epochs'):
        state = env.reset()
        for t in tqdm(range(X.shape[0]),desc='iterations'):  # Don't infinite loop while learning
#            print(t,t%1000==0)
            action = select_action(state)
#            print(state.size(),action.size())
            state, reward, = env.step(action)
#            if True:
#                env.render()
            policy.rewards.append(reward)
#            if done:
#                break
#            if t%100==0:print('guru')
#            print('ch')
            if ((t+1)%1000)==0:
#                print('back-prop')
                finish_episode(i_episode)
#                print(list(policy.named_parameters()))
        running_reward = running_reward * 0.99 + t * 0.01
        for idx,p in enumerate(policy.parameters()):
            writer.add_histogram(f'model_parameters_{idx}',p.grad.detach().numpy())
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
#        if running_reward > env.spec.reward_threshold:
#            print("Solved! Running reward is now {} and "
#                  "the last episode runs to {} time steps!".format(running_reward, t))
#            break

def save_model():
    torch.save(policy.state_dict(),'models\policy.pth')
#    torch.save
def validate(X,Y):
    pred=policy(torch.FloatTensor(X))
    df=pd.DataFrame()
    df['act']=Y
    df['pred']=Categorical(pred).sample().detach().numpy()
    print("results")
    print(np.mean(df['pred'].values==y))
    return df
if __name__ == '__main__':
    #data=pd.read_feather(r'tagged_data_feather')
    #cont_names=['BerPreFecMax','PhaseCorrectionAve','PmdMin','Qmin','SoPmdAve']
    #X,y=data[cont_names].copy(),data['tagged_alerts'].copy()
    try:
        data=pd.read_csv(r'yahoo_1980_2018.csv')
#        X,y=dummy_data=make_classification(n_classes=2,n_samples=10000,n_features=5,)
#        X,y=make_imbalance(X,y,{0:4000,1:500})
        X,y=data.loc[:,'Close'].values.copy(),data.iloc[:,-1].values.copy()
        print(X.shape,y.shape)
#        print(X.head())
        X=X.reshape(-1,1)
        anomdataset=AnomDataset(X,y)
        data_loader=DataLoader(anomdataset,batch_size=10)    
        env=AnomalyEnv(data_loader,writer)
        main()
        save_model()
        df=validate(X,y)
        df.to_csv('val.csv')
         
    except KeyboardInterrupt:
        print('Want to save your model')
        inp=input('Y/N')
        if inp=='Y':
            save_model()
            validate(X,y)
        else:
            sys.exit()
     

#x1=x.view(-1,1,1)
#q=conv(x1)

#import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
#pca=PCA(n_components=2)     
#pcaC=pca.fit_transform(X)     
#plt.scatter(pcaC[:,0],pcaC[:,1],c=y)     
#plt.show()
#     
     
     
     