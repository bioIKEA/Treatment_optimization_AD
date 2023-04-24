import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

##setting up display limits for rows and columns 
pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 20)

whole_data_without_states= pd.read_csv("/Users/kritibbhattarai/Desktop/Treatment_optimization_AD/revision/codes/whole-data/data/whole-data-without-states.csv")

whole_data_without_states=whole_data_without_states.dropna(subset=['CMREASON'])

whole_data_without_states['CMMED'] = whole_data_without_states['CMMED'].apply(literal_eval)

def check_group(group):
    # print(group)
    ad_checker=False
    hyp_checker=False
    if isinstance(group['CMREASON'], pd.Series) or isinstance(group['CMMED'], pd.Series):
        for row_id, row in group.iterrows():
            if ('ad' not in row['CMREASON'] and 'inhibitors' not in row['CMMED'] and 'namenda' not in row['CMMED']) and ('depression' not in row['CMREASON'] and 'depression' not in row['CMMED']) and ('hypertension' in row['CMREASON'] or 'hypertension' in row['CMMED']):
                hyp_checker=True
            else:
                break
        if hyp_checker:
            return group
            # if ('ad' in row['CMREASON'] or 'inhibitors' in row['CMMED'] or 'namenda' in row['CMMED']):
            #     ad_checker=True
            # if ('depression' in row['CMREASON'] or 'depression' in row['CMMED']):
            #     break
            # if ('hypertension' in row['CMREASON'] and 'hypertension' in row['CMMED']): 
            #     hyp_checker=True
            # if ('ad' in row['CMREASON'] or 'inhibitors' in row['CMMED'] or 'namenda' in row['CMMED']) or ('hypertension' in row['CMREASON'] or 'hypertension' in row['CMMED']): 
            # if ('ad' in row['CMREASON'] or 'inhibitors' in row['CMMED'] or 'namenda' in row['CMMED']) and ('depression' not in row['CMREASON'] and 'depression' not in row['CMMED']) and ('hypertension' in row['CMREASON'] or 'hypertension' in row['CMMED']) :
            # if ('hypertension' in row['CMREASON'] or 'hypertension' in row['CMMED']): 
        # if ad_checker and  hyp_checker:
        #     return group
ad_hyp_data= whole_data_without_states.groupby('RID').apply(check_group)
print(ad_hyp_data.head(5))

ad_hyp_data= ad_hyp_data.reset_index(drop=True)

#printing the dimension of data and all column names

print("\ndata dimension= ", ad_hyp_data.shape)
# print("\ncolumn_names= ", ad_hyp_data.columns)

##demographics

#gender

male_female=ad_hyp_data.groupby(['PTGENDER'])['RID'].nunique()
print("\n", male_female)

###total patients

total_patients=len(pd.unique(ad_hyp_data['RID']))
print("\nTotal Patients= ",total_patients)

#total visits

total_visits=ad_hyp_data['RID'].value_counts()
print("\nMean Total Visits= ", total_visits.mean())
print("\nSD Total Visits= ", total_visits.std())



#monthly visits

monthly_visits=ad_hyp_data.groupby(['RID'])['Month'].max()
print("\nMean Monthly Visits= ", monthly_visits.mean())
print("\nSD Monthly Visits= ", monthly_visits.std())



print(ad_hyp_data.shape)
# print(ad_hyp_data.head(500))


# ad_hyp_data.to_csv('ad_hyp_data_without_states.csv', index = False)  # Export merged pandas DataFrame



