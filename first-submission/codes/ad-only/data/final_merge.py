import pandas as pd
import numpy as np
from ast import literal_eval

pd.set_option('display.max_rows', 500)

merged_file= pd.read_csv("/Users/kritibbhattarai/Desktop/internship/Alzheimer's/python/new/ad-only/data/merged_wo-sa.csv")

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
    if ( row['ADAS13_pre'] <= 25.164999961853027 ):
        if ( row['ADAS13_pre'] <= 18.5 ) :
            if ( row['RAVLT.learning_pre'] <= 1.5 ) :
                state=0
            else :
                state=1
        else :
            if ( row['MOCA_pre'] <= 21.5 ) :
                if ( row['FDG_pre'] <= 5.991738796234131 ) :
                    state=2
                else :
                    state=3
            else:
                    state=4
    else:
        if ( row['ADAS13_pre'] <= 35.834999084472656 ):
            if ( row['CDRSB_pre'] <= 3.25 ) :
                    state=5
            else :
                if ( row['FDG_pre'] <= 5.047677278518677 ) :
                    state=6
                else :
                    state=7
        else :
                    state=8

    state_list.append(state)
merged_file = merged_file.assign(states= state_list)







merged_data=merged_file
merged_data['CMMED'] = merged_data['CMMED'].apply(literal_eval)

def check_bl(row):
        # print(row)
        if row['VISCODE']=='bl' and set(['-4'])==set(row['CMMED']):
                return False
        return True
merged_data= merged_data[merged_data.apply(check_bl, axis=1)]


###adding a new action column where {no drugs:0,inhibitors:1,namenda:2,hypertension:3,inhibitors+namenda:4,inhibitors+hypertension:5,namenda+hypertension:6,inhibitors+hypertension+namenda:7}


action_list=[]
action=0
for med_list in merged_data['CMMED']:
    if set(['inhibitors', 'namenda']).issubset(set(med_list)):
        action=4
        # print("four",med_list)
    elif set(['hypertension']).issubset(set(med_list)):
        action=3
        # print("three",med_list)
    elif set([ 'namenda']).issubset(set(med_list)):
        # print("two",med_list)
        action=2
    elif set(['inhibitors']).issubset(set(med_list)):
        action=1
        # print("one",med_list)
    elif set(['-4'])==set(med_list):
        action=0
    else:
        action=5
    action_list.append(action)
merged_data2= merged_data.assign(action = action_list)
print(merged_data2['action'].value_counts())

print("\nmerged data dimension= ", merged_data2.shape)


merged_data2.to_csv('merged_finalMMSE.csv', index = False)  # Export merged pandas DataFrame



