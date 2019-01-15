# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:27:03 2018

@author: gurunath.lv
"""

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
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from torch.autograd import Variable
from imblearn.datasets import make_imbalance
from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np
import pickle
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
#class AnomDataset(Dataset):
#    def __init__(self,X,y):
#        self.X=X
#        self.Y=y
#
#    def __getitem__(self,index):
#        return Variable(torch.FloatTensor(self.X[index])),Variable(torch.LongTensor([self.Y[index]]))
#    def __len__(self):
#        return self.X.shape[0]


class ForDataset(Dataset):
    def __init__(self,X):        
        self.X,self.Y=self.prepare_train_data(X)
    def __getitem__(self,index):
        return self.X[index],self.Y[index]
    def __len__(self):
        return self.X.shape[0]
    def prepare_train_data(self,X,train_len=96,pred_len=4):
        X=self.transformer(X)
        np_x=[]
        np_y=[]
        for i in tqdm(range(0,X.shape[0]-train_len-pred_len-3),desc='Prepare_data'):
#            if i%96==0:
#                print(X[i:train_len+i].reshape(1,-1).shape,X[i+train_len+pred_len:i+train_len+pred_len+4].reshape(1,-1).shape)
            np_x.append(X[i:train_len+i].reshape(1,-1))
            np_y.append(X[i+train_len+pred_len:i+train_len+pred_len+4].reshape(1,-1))
        return np.vstack(np_x),np.vstack(np_y)
    def transformer(self,data,name_to_save='yahoo_scaler'):
        scaler = Normalizer()
        scaled_out=scaler.fit_transform(data)
#        print(scaler.data_min_,scaler.data_max_)
        pickle.dump(scaler,open(f'{name_to_save}.pkl','wb'))
    
        return scaled_out

 
def get_data():
    import pandas_datareader.data as web
    import datetime
    
    start = datetime.datetime(1980, 1, 1)
    
    end = datetime.datetime(2018, 10, 5)
    
    f = web.DataReader('F', 'yahoo', start, end,)
    return f    
       
#data.to_csv('yahoo_1980_2018.csv')
data=get_data()
data.Close.plot()
data['Anom']=np.zeros_like(data.iloc[:,0])
plt.plot(data.index,[data.Close.mean()]*data.shape[0])
close=np.abs(data.Close-data.Close.mean())
anom_mask=close>1.5*data.Close.std()
data.loc[anom_mask[anom_mask.values].index,'Anom']=1
anom=data.Close[data['Anom'].values==1]
plt.scatter(anom.index,anom.values,color='red')
plt.show()

fordataset=ForDataset(data.iloc[:,:-1])
dataloader=DataLoader(fordataset,batch_size=10)                
s=iter(dataloader)
x,y=next(s)
#dummy_data=make_classification(n_classes=2,n_samples=10000,n_features=5,)
#state,reward=aenv.step(y)
#aenv.render()

fordataset=ForDataset(data['Close'].values.reshape(-1,1))
dataloader=DataLoader(fordataset,batch_size=10)                
s=iter(dataloader)
x,y=next(s)


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
        return action_scores


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
#    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
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
    i=0
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        loss=-log_prob * reward
        policy_loss.append(loss)
        writer.add_scalar('trainloss',loss.item(),episode+i)
        i+=1
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in tqdm(range(10),desc='Epochs'):
        state = env.reset()
        for t in tqdm(range(10),desc='iterations'):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
#            if done:
#                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode(i_episode)
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
    data=pd.read_feather(r'tagged_data_feather')
    cont_names=['BerPreFecMax','PhaseCorrectionAve','PmdMin','Qmin','SoPmdAve']
    X,y=data[cont_names].copy(),data['tagged_alerts'].copy()
    X,y=make_imbalance(X,y,{0:4000,1:500})
    anomdataset=AnomDataset(X,y)
    data_loader=DataLoader(anomdataset,batch_size=1)    
    env=AnomalyEnv(data_loader,writer)
    main()
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     