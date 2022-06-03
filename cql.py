import pandas as pd
import numpy as np
import random
from random import shuffle
from collections import Counter
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQL
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split

pd.options.display.float_format = '{:.2f}'.format

data_path="/Users/kritibbhattarai/Desktop/internship/Alzheimer's/python/"

data1= pd.read_csv(data_path+'merged.csv', low_memory=False)

# print("data shape= ", data.shape)

# print("column 98= ", data.columns[98])

taken =['RID', 'AGE','CDRSB','MMSE', 'ADAS13','MOCA','CMMED','action','states']

data = data1.loc[:, taken]
# print("data \n", data)

# print("unique values count for CMMED",data.CMMED.value_counts())

##fill empty ADAS13 and MOCA score
data['ADAS13']=data.groupby('RID')['ADAS13'].ffill().bfill()
data['MOCA']=data.groupby('RID')['MOCA'].ffill().bfill()


##################################################################test section#########################################################################






##################################################################test section#########################################################################



data['terminal']=0

# print(len(data.index))

for ind in data.index:
  if ind+1<len(data.index):
     if (data['RID'][ind] != data['RID'][ind+1]):
         data.loc[ind,'terminal']=1
  else:
         data.loc[ind,'terminal']=1

# print(data.terminal.value_counts())



obs =['AGE', 'CDRSB', 'MMSE', 'ADAS13', 'MOCA']

data['reward']=0


for ind in data.index[1:]:
  if (data['terminal'][ind-1]!=1):
     data.loc[ind,'reward']= data.loc[ind,'MMSE'] - data.loc[ind-1,'MMSE']

# print(data.head(31))

observations= data.loc[:, obs].to_numpy()
actions = data.loc[:, 'action'].to_numpy()
rewards = data.loc[:, 'reward'].to_numpy()
terminals = data.loc[:, 'terminal'].to_numpy()

# print(np.any(np.isnan(observations)))

dataset = MDPDataset(observations, actions, rewards, terminals)

# automatically splitted into d3rlpy.dataset.Episode objects
dataset.episodes

# each episode is also splitted into d3rlpy.dataset.Transition objects
episode = dataset.episodes[0]
episode[0].observation
episode[0].action
episode[0].reward
episode[0].next_observation
episode[0].terminal

# d3rlpy.dataset.Transition object has pointers to previous and next
# transitions like linked list.
transition = episode[0]
while transition.next_transition:
    transition = transition.next_transition

# save as HDF5
dataset.dump('dataset.h5')

# load from HDF5
new_dataset = MDPDataset.load('dataset.h5')


### setup CQL algorithm
cql = DiscreteCQL(use_gpu=False, learning_rate=0.05, target_update_interval=100)

### split train and test episodes
train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

### start training
cql.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs=100,
        scorers={'td_error': td_error_scorer,'value_scale': average_value_estimation_scorer})

td_error = td_error_scorer(cql, test_episodes)

print(td_error)
