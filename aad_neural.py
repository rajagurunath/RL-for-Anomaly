import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset,sampler
df=pd.read_csv(r'yahoo_1980_2018.csv')

def pretraining_annotation(df,alpha=3):
    df['tag']=np.zeros(df.shape[0],dtype='int')
    for col,mean in zip(df.iloc[:,:-1].mean().index,df.iloc[:,:-1].mean()):
        print(col,mean)
        df['tag'][(df[col]-mean-alpha*df[col].std())>0]=1
    print('Class Count:',df['tag'].value_counts())
    return df
df=pretraining_annotation(df)

class AnnotSet(Dataset):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self,ind):
        return self.X[ind],self.Y[ind]

x_columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
annotset=AnnotSet(df[x_columns].values,df['tag'].values)
class_sample_count=df['tag'].value_counts().values
class_weights=1./torch.from_numpy(class_sample_count.astype('float'))
sampler=torch.utils.data.sampler.WeightedRandomSampler(class_weights,10,replacement=True)
annoloader=DataLoader(annotset,batch_size=10,sampler=sampler)

al=iter(annoloader)
x,y=next(al)

class contLearner(nn.Module):
    def __init__(self,inp_sz,emb_sz:LISTofDict,hdn_sz,out_sz):
        super(self,contLearner)__init__()
        self.emb_layers=nn.Embedding()
        self.dense=nn.Linear(inp_sz,hdn_sz)







