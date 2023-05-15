import numpy as np
import pandas as pd
import random
from random import shuffle
from collections import Counter
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
data_path="/Users/kritibbhattarai/Desktop/internship/Alzheimer's/python/new/whole-data/data/"

num_states=13
num_actions=6
epsilon=0.01/(num_actions-1)
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
behavior_policy = np.ones((num_states, num_actions))*epsilon
groups12 = data.groupby('states')
for state, rows in groups12:
    cnt = Counter(rows['action'])
    for action, count in cnt.items():
        behavior_policy[int(state), int(action)]=count
# print(behavior_policy.sum(axis=1)[:, np.newaxis])
# print(behavior_policy)
df1 = pd.DataFrame(behavior_policy, columns = ['No','In','Na','Hy', 'Ni', 'So'], index = ['S0','S1', 'S2','S3', 'S4','S5','S6', 'S7','S8', 'S9','S10', 'S11','S12'])
print(df1)

ai_policy_array=np.array([[1, float('nan'), float('nan'),float('nan'),float('nan'),float('nan')], [float('nan'), float('nan'), float('nan'),float('nan'),float('nan'),1],[1, float('nan'), float('nan'),float('nan'),float('nan'),float('nan')],[float('nan'), float('nan'), float('nan'),float('nan'),float('nan'),1],[float('nan'), 1, float('nan'),float('nan'),float('nan'),float('nan')],[float('nan'), float('nan'), float('nan'),1,float('nan'),float('nan')],[float('nan'), float('nan'), 1,float('nan'),float('nan'),float('nan')],[float('nan'), float('nan'), float('nan'),1,float('nan'),float('nan')],[float('nan'), float('nan'), float('nan'),float('nan'),float('nan'),1],[1, float('nan'), float('nan'),float('nan'),float('nan'),float('nan')],[float('nan'), float('nan'), float('nan'),1,float('nan'),float('nan')],[float('nan'), float('nan'), float('nan'),float('nan'),1,float('nan')],[float('nan'), 1, float('nan'),float('nan'),float('nan'),float('nan')]])

df2 = pd.DataFrame(ai_policy_array, columns = ['No_ai','In_ai','Na_ai','Hy_ai', 'Ni_ai', 'So_ai'], index = ['S0','S1', 'S2','S3', 'S4','S5','S6', 'S7','S8', 'S9','S10', 'S11','S12'])
print(df2)
frames = [df1, df2]
df= df1.join(df2)
df=df.reset_index()
print(df)

fig=df.plot(x="index", kind="bar", stacked=True, title="States Actions Count", figsize=(15, 10))
plt.ylim(0,900)
# fig.set_figheight(10)
# fig.set_figwidth(15)
plt.xlabel("states")
plt.ylabel("counts")

plt.savefig('barplot.png')
plt.show()
# fig.show()

