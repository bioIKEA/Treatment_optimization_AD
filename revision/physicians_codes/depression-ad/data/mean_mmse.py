import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data= pd.read_csv('/Users/kritibbhattarai/Desktop/Treatment_optimization_AD/physicians_codes/depression-ad/data/merged_wo-sa.csv')


data['avg_mmse']=data.groupby(['RID'])['MMSE'].transform('mean')

#finding mean of avg_mmse column
new_data=data.groupby(['RID','avg_mmse'])['avg_mmse'].mean().reset_index(name='mean')
mean_mmse=new_data['mean'].mean()
print("mean of avg_mmse column: ", new_data['mean'].mean())

#filtering groups
data_sr=data.groupby('RID').filter(lambda x: (x.avg_mmse>=mean_mmse).any())
data_jr=data.groupby('RID').filter(lambda x: (x.avg_mmse<mean_mmse).any())
# print(data.head(100))


data_sr.to_csv('./sr/sr_phy_wo_sa.csv', index = False)  # Export merged pandas DataFrame
data_jr.to_csv('./jr/jr_phy_wo_sa.csv', index = False)  # Export merged pandas DataFrame
