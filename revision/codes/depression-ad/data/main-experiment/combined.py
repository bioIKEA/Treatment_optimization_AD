import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

##setting up display limits for rows and columns 
pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 20)


whole_data_without_states= pd.read_csv("/Users/kritibbhattarai/Desktop/Treatment_optimization_AD/revision/codes/whole-data/data/main-experiment/whole-data-without-states.csv")
# print(whole_data_without_states.head(5))

whole_data_without_states['CMREASON']=whole_data_without_states['CMREASON'].fillna("['-4']")
whole_data_without_states['CMMED']=whole_data_without_states['CMMED'].fillna("['-4']")

whole_data_without_states['CMMED'] = whole_data_without_states['CMMED'].apply(literal_eval)
whole_data_without_states['CMREASON'] = whole_data_without_states['CMREASON'].apply(literal_eval)

def check_group(group):
    if isinstance(group['CMREASON'], pd.Series) or isinstance(group['CMMED'], pd.Series):
        for row_id, row in group.iterrows():
            if ('ad' in row['CMREASON'] or 'inhibitors' in row['CMMED'] or 'namenda' in row['CMMED']) or ('depression' in row['CMREASON'] or 'depression' in row['CMMED']):
                return group
ad_depression_data= whole_data_without_states.groupby('RID').apply(check_group)
# print(ad_depression_data.head(5))

ad_depression_data= ad_depression_data.reset_index(drop=True)

#printing the dimension of data and all column names

print("\ndata dimension= ", ad_depression_data.shape)
print("\ncolumn_names= ", ad_depression_data.columns)

##demographics

#gender

male_female=ad_depression_data.groupby(['PTGENDER'])['RID'].nunique()
print("\n", male_female)

###total patients

total_patients=len(pd.unique(ad_depression_data['RID']))
print("\nTotal Patients= ",total_patients)

#total visits

total_visits=ad_depression_data['RID'].value_counts()
print("\nMean Total Visits= ", total_visits.mean())
print("\nSD Total Visits= ", total_visits.std())



#monthly visits

monthly_visits=ad_depression_data.groupby(['RID'])['Month'].max()
print("\nMean Monthly Visits= ", monthly_visits.mean())
print("\nSD Monthly Visits= ", monthly_visits.std())



print(ad_depression_data.shape)


ad_depression_data.to_csv('ad_depression_data.csv', index = False)  # Export merged pandas DataFrame



