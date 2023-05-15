import pandas as pd
import numpy as np
from ast import literal_eval

pd.set_option('display.max_rows', 500)

merged_file= pd.read_csv("/Users/kritibbhattarai/Desktop/Treatment_optimization_AD/revision/codes/ad-only/data/main-experiment/ad_data_without_states.csv")

#printing the dimension of data and all column names

print("\ndata dimension= ", merged_file.shape)
print("\ncolumn_names= ", merged_file.columns)

##demographics

#gender

male_female=merged_file.groupby(['PTGENDER'])['RID'].nunique()
print("\n", male_female)

###total patients

total_patients=len(pd.unique(merged_file['RID']))
print("\nTotal Patients= ",total_patients)

#total visits

total_visits=merged_file['RID'].value_counts()
print("\nMean Total Visits= ", total_visits.mean())
print("\nSD Total Visits= ", total_visits.std())



#monthly visits

monthly_visits=merged_file.groupby(['RID'])['Month'].max()
print("\nMean Monthly Visits= ", monthly_visits.mean())
print("\nSD Monthly Visits= ", monthly_visits.std())

############assigining_states####################

state=0
state_list=[]
for index,row in merged_file.iterrows():
    if ( row['ADAS13_pre'] <= 25.835000038146973 ) :
        if ( row['ADAS13_pre'] <= 22.164999961853027 ) :
            if ( row['MOCA_pre'] <= 20.5 ) :
                if ( row['CDRSB_pre'] <= 4.75 ) :
                    state=0
                else :
                    state=1
            else :
                state=2
        else :
            if ( row['MOCA_pre'] <= 19.5 ) :
                state=3
            else :
                state=4
    else :
        if ( row['ADAS13_pre'] <= 35.834999084472656 ) :
            if ( row['CDRSB_pre'] <= 3.75 ) :
                state=5
            else :
                state=6
        
        else :
            if ( row['MOCA_pre'] <= 12.5 ) :
                state=7
            
            else :
                state=8
    state_list.append(state)
merged_file = merged_file.assign(states= state_list)
merged_data=merged_file
merged_data['CMMED']=merged_data['CMMED'].apply(lambda x:"[\"-4\"]" if pd.isna(x) else x)
merged_data['CMMED'] = merged_data['CMMED'].apply(literal_eval)


###adding a new action column where {no drugs:0,inhibitors:1,namenda:2,hypertension:3,inhibitors+namenda:4,inhibitors+hypertension:5,namenda+hypertension:6,inhibitors+hypertension+namenda:7}


action_list=[]
action=0
for med_list in merged_data['CMMED']:
    if 'inhibitors' in med_list and 'namenda' in med_list:
        action=4
    elif 'inhibitors' in med_list:
        action=1
    elif 'namenda' in med_list:
        action=2
    elif 'hypertension' in med_list:
        action=3
    else:
        action=0
    action_list.append(action)

merged_data2= merged_data.assign(action = action_list)
print(merged_data2['action'].value_counts())

print("\nmerged data dimension= ", merged_data2.shape)


merged_data2.to_csv('ad_data_with_states.csv', index = False)  # Export merged pandas DataFrame



