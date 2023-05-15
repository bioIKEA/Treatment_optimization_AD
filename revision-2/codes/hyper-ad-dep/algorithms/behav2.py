import numpy as np
import pandas as pd
import random
from random import shuffle
from collections import Counter

pd.options.display.float_format = '{:.2f}'.format


#######################################################################algorithms#####################################################################


##################################################################initialization##########################################################

num_states=10
num_actions=6
epsilon=0.01/(num_actions-1)
#discount factor
gamma=0.3
#theta
theta=0.0001

######################################testing######################################
def weighted_importance_sampling(policy, test_set, behavior_policy):
    
    test_set['timestep']=''
    test_set['rho']=''
    
    groups = test_set.groupby('RID')
    for rid, rows in groups:
        timestep=0
        rho_cum=1
        #each_row
        for index, row in rows.iterrows():
            state=int(row['states'])
            action=int(row['action'])
            test_set.loc[index, 'timestep']=timestep
            rho_cum = rho_cum*policy[state,action]/behavior_policy[state, action]
            #test_set.iloc[i,7]=rho_cum
            test_set.loc[index, 'rho']=rho_cum

            timestep = timestep+1
            
            

    values=[]
    value_sum =0
    for rid, rows in groups:
        value_sum =0
        sum_of_rho=rows['rho'].sum()
        for index, row in rows.iterrows():
            reward = row['reward']
            timestep= row['timestep']
            rho=row['rho']
            value_sum = value_sum + (gamma**timestep)*reward*rho/sum_of_rho
        values.append(value_sum)
    # print(value_sum)    
    return(value_sum)
    # return(values)
    

####################################################################Q_learning#######################################################################
alpha=0.05

def derive_policy(q, epsilon=epsilon):
    policy = np.ones((num_states, num_actions))*0.0000000001
    for state in range(num_states):
        best_action = np.argmax(q[state,:])
        policy[state, best_action]=1-num_actions*0.0000000001
    return policy


def QLearning(q_table, train_data):

    groups=train_data.groupby('RID')
    for rid, rows in groups:
        for index, row in rows.iterrows():
            state=int(row['states'])
            action=int(row['action'])
            reward = row['reward']
            try:
                next_state = rows.loc[index+1, 'states']
                # max_action_value = q_table[next_state,np.argmax(q_table[next_state])]
                max_action_value=np.max(q_table[next_state, :])
                
                q_table[state, action] = q_table[state, action] + \
                            alpha * (reward + gamma * max_action_value - q_table[state,action])
                
                
            except KeyError:
                pass
    return q_table



######################################################################################################################mainsplit###############################################




data_path="/Users/kritibbhattarai/Desktop/internship/Alzheimer's/python/new-w-bootstrap/hyper-ad-dep-states/data/"

data1= pd.read_csv(data_path+'merged_final.csv', low_memory=False)

# print("data shape= ", data1.shape)

# print("column 98= ", data.columns[98])

taken =['RID', 'AGE','CDRSB','MMSE', 'ADAS13','MOCA','CMMED','action','states']

data = data1.loc[:, taken]
# print("data \n", data)

# print("unique values count for CMMED",data.CMMED.value_counts())

##fill empty ADAS13 and MOCA score
data['ADAS13']=data.groupby('RID')['ADAS13'].ffill().bfill()
data['MOCA']=data.groupby('RID')['MOCA'].ffill().bfill()

data['terminal']=0

# print(len(data.index))

for ind in data.index:
  if ind+1<len(data.index):
     if (data['RID'][ind] != data['RID'][ind+1]):
         data.loc[ind,'terminal']=1
  else:
         data.loc[ind,'terminal']=1

# print(data.terminal.value_counts())


obs=['states']
data['reward']=0



# for ind in data.index[0:]:
#   if (data['terminal'][ind]!=1):
#      data.loc[ind,'reward']= data.loc[ind,'ADAS13']- data.loc[ind+1,'ADAS13']


for ind in data.index[1:]:
  if (data['terminal'][ind-1]!=1):
     data.loc[ind,'reward']= data.loc[ind,'MMSE']- data.loc[ind-1,'MMSE']

#####################################splitting data###########################################
#split into train/validation/test

random.seed(42)
unique_ids = data.RID.unique()
random.shuffle(unique_ids)
train_sample=0.8
test_sample=0.2
train_num = int(len(unique_ids) * train_sample) 
train_ids = unique_ids[:train_num]
test_ids = unique_ids[train_num:]

train_set = pd.DataFrame()
train_set = data.loc[data['RID'].isin(train_ids)].reset_index(drop=True)

test_set = pd.DataFrame()
test_set = data.loc[data['RID'].isin(test_ids)].reset_index(drop=True)


old_q_reward=-100

for internal_repitition in range(50):

    random.seed(42)
    unique_train_ids=train_set.RID.unique()
    random.shuffle(unique_train_ids)
    full_train_sample=0.8
    val_sample=0.2
    full_train_num = int(len(unique_ids)*full_train_sample) 
    val_num = int(len(unique_ids)*val_sample) 
    full_train_ids = unique_ids[:train_num]
    val_ids=unique_train_ids[-val_num:]

    full_train_set = pd.DataFrame()
    full_train_set = data.loc[data['RID'].isin(full_train_ids)].reset_index(drop=True)

    val_set = pd.DataFrame()
    val_set = data.loc[data['RID'].isin(val_ids)].reset_index(drop=True)



    ##########behavior_policy#########

    behavior_policy = np.ones((num_states, num_actions), dtype=np.float32)*epsilon
    groups12 = full_train_set.groupby('states')
    for state, rows in groups12:
        cnt = Counter(rows['action'])
        for action, count in cnt.items():
            behavior_policy[int(state), int(action)]=count
    # print(behavior_policy.sum(axis=1)[:, np.newaxis])
    behavior_policy=behavior_policy/behavior_policy.sum(axis=1)[:, np.newaxis]
    
    ##########q_learning########

    q1=np.zeros((num_states, num_actions), dtype=np.float32)
    q_tables=QLearning(q1, full_train_set)
    q_policy=derive_policy(q_tables, epsilon)

    current_q_reward=weighted_importance_sampling(q_policy, val_set,behavior_policy)
    if current_q_reward>=old_q_reward:
        best_q_policy= q_policy
        old_q_reward=current_q_reward
states_counts=data.groupby('states')['states'].count().reset_index(name='counts')
# print(states_counts)
df2=best_q_policy
# df2 = pd.DataFrame(best_q_policy, columns = ['No','In','Na','Hy', 'Ni', 'So'], index = ['S0','S1', 'S2','S3', 'S4','S5','S6', 'S7','S8', 'S9']).reset_index()
df2 = pd.DataFrame(best_q_policy, columns = ['No_ai','In_ai','Na_ai','Hy_ai', 'Ni_ai', 'So_ai'])
# print(df2)
# print(states_counts['counts'].squeeze())

# print(type(states_counts['counts'].squeeze()))
df3=df2.mul(states_counts['counts'].squeeze(), axis=0)
value=['S0','S1', 'S2','S3', 'S4','S5','S6', 'S7','S8', 'S9']
df3.insert(loc=0,column='index', value=value)
# print(df3)
##########behavior_policy#########

behavior_policy = np.ones((num_states, num_actions), dtype=np.float32)*epsilon
groups12 = data.groupby('states')
for state, rows in groups12:
    cnt = Counter(rows['action'])
    # print(cnt)
    for action, count in cnt.items():
        # print(action)
        behavior_policy[int(state), int(action)]=count
# print(behavior_policy.sum(axis=1)[:, np.newaxis])
# behavior_policy=behavior_policy/behavior_policy.sum(axis=1)[:, np.newaxis]
    

# print(behavior_policy)

df1 = pd.DataFrame(behavior_policy, columns = ['No','In','Na','Hy', 'Ni', 'So'], index = ['S0','S1', 'S2','S3', 'S4','S5','S6', 'S7','S8', 'S9'])
df1=df1.reset_index()
# print(df1)
df = df1.merge(df3, on='index', how='left')
print(df)
df_transposed=df.set_index('index').T
print(df_transposed)
print(df_transposed.columns)
d={}
for i in range(10):
    row=df.iloc[[i]]
    # print(row)
    # print(row['No'].values[0])
    d["data_s{0}".format(i)] = pd.DataFrame([
        ('No','data',row['No'].values[0]),
        ('No','AI',row['No_ai'].values[0]),
        ('In','data',row['In'].values[0]),
        ('In','AI',row['In_ai'].values[0]),
        ('Na','data',row['Na'].values[0]),
        ('Na','AI',row['Na_ai'].values[0]),
        ('Hy','data',row['Hy'].values[0]),
        ('Hy','AI',row['Hy_ai'].values[0]),
        ('Ni','data',row['Ni'].values[0]),
        ('Ni','AI',row['Ni_ai'].values[0]),
        ('So','data',row['So'].values[0]),
        ('So','AI',row['So_ai'].values[0]),
    ], 
    columns=['action', 'type', 'value']
).set_index(['action','type']).value



import matplotlib.pyplot as plt
# fig=df.plot(x="index", kind="bar", stacked=True, title="States Actions Count", figsize=(15, 10))
df.set_index('index',inplace=True)
for counter in range(10):
    ax=plt.subplot(5,2,counter+1)
    fig=d["data_s{0}".format(counter)].unstack().plot(kind="bar",  title=f"State{counter} Actions Counts", figsize=(15, 10),ax=ax)
    plt.tight_layout(pad=2)
    plt.xlabel("actions")
    plt.ylabel("counts")

    # plt.show()
    plt.savefig(f'barplot_state{counter}.png')
    # fig.clf()
    # fig.show()

