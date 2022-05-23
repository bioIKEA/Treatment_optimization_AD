import gym
import os
import numpy as np
import parl
from parl.utils import logger, ReplayMemory
from cartpole_model import CartpoleModel
from cartpole_agent import CartpoleAgent
from parl.algorithms import DQN
import pandas
import csv, operator
import environment

#LEARN_FREQ = 5  # training frequency
#MEMORY_SIZE = 200000
#MEMORY_WARMUP_SIZE = 200
#BATCH_SIZE = 64
#LEARNING_RATE = 0.0005
#GAMMA = 0.99
# env = AlzheimersEnv()
# print('print(env.action_space)')
#print(env.action_space)
#print("print(env.action_space.n)")
#print(env.action_space.n)
##> Discrete(2)
#print("print(env.observation_space)")
#print(env.observation_space)
#print("print(env.observation_space.shape)")
#print(env.observation_space.shape)
#print("print(env.observation_space.shape[0])")
#print(env.observation_space.shape[0])
#obs_dim = env.observation_space.shape[0]
#act_dim = env.action_space.n

#rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 0)
#print(len(rpm))

#model = CartpoleModel(obs_dim=obs_dim, act_dim=act_dim)
#print(model)
#alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
#print(alg)
#agent = CartpoleAgent( alg, act_dim=act_dim, e_greed=0.1, e_greed_decrement=1e-6)
#print(agent)
##> Box(4,)

file = open('/Users/kritibbhattarai/Desktop/Combined.csv')
csvreader = csv.reader(file, delimiter=',')
reader=next(csvreader)
# print(reader)
# csvreader=sorted(csvreader, key=operator.itemgetter(0))
# print(csvreader[0:100])
# print(reader.index('RID'))
# print(reader.index('MMSE'))
# print(reader.index('MOCA'))
# print(reader.index('ADAS13'))
# print(reader.index('CMMED'))
lis=[]
dic={}
# counter=15
for row in csvreader:
    #RID
    id_1=float(row[0])
    #MMSE
    mmse=float(row[21])
    #ADAS13
    ADAS13=float(row[20])
    #action
    action=float(row[98])
    # if counter>0:

    if id_1 in dic:
        dic[id_1].append((mmse,ADAS13,action))
        # print("dic",dic)
    else:
        dic={}
        dic[id_1]=([(mmse,ADAS13,action)])
        
        lis.append(dic)
        # print("lis", lis)
    # counter-=1
print(lis[25])
print(lis[24])
# for i in range(len(lis)):
#     print(i)
#     a=list(lis[i].keys())[0]
#     if len(lis[i][a])==0:
#         print (lis[i])
#         break
#print(list(lis[0].keys())[0])
#a=lis[0]
#print(a[2])
#print(a[2][0])
#normal_ADAS13, normal_MMSE,_=a[2][0]
#print(normal_ADAS13)
#print(normal_MMSE)
## print(len(lis))
## print(lis[0])
