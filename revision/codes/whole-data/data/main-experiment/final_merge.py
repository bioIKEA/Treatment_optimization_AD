import pandas as pd
import numpy as np
from ast import literal_eval

pd.set_option('display.max_rows', 500)
merged_file= pd.read_csv("/Users/kritibbhattarai/Desktop/Treatment_optimization_AD/revision/codes/whole-data/data/main-experiment/whole-data-without-states.csv")
print(merged_file.shape)

############assigining_states####################

state=0
state_list=[]
for index,row in merged_file.iterrows():
    if ( row['ADAS13_pre'] <= 25.835000038146973 ) :
        if ( row['CDRSB_pre'] <= 3.75 ) :
            if ( row['RAVLT.immediate_pre'] <= 27.5 ) :
                if ( row['ADAS13_pre'] <= 22.164999961853027 ) :
                    state=0
                else :
                    state=1
            else :
                if ( row['ADAS13_pre'] <= 21.164999961853027 ) :
                    state=2
                else :
                    state=3
        else :
            if ( row['MOCA_pre'] <= 19.5 ) :
                if ( row['CDRSB_pre'] <= 4.75 ) :
                    state=4
                else :
                    state=5
            else :
                state=6
    else :
        if ( row['ADAS13_pre'] <= 35.5 ) :
            if ( row['CDRSB_pre'] <= 3.75 ) :
                if ( row['CDRSB_pre'] <= 2.0 ) :
                    state=7
                else :
                    state=8
            else :
                state=9
        else :
            if ( row['ADAS13_pre'] <= 52.165000915527344 ) :
                if ( row['MOCA_pre'] <= 15.5 ) :
                    state=10
                
                else :
                    state=11
            else :
                state=12
    state_list.append(state)
merged_file = merged_file.assign(states= state_list)

merged_data=merged_file
merged_data['CMMED']=merged_data['CMMED'].apply(lambda x:"[\"-4\"]" if pd.isna(x) else x)
merged_data['CMMED'] = merged_data['CMMED'].apply(literal_eval)

# def check_bl(row):
#         # print(row)
#         if row['VISCODE']=='bl':
#                 return False
#         return True
# merged_data= merged_data[merged_data.apply(check_bl, axis=1)]


###adding a new action column where {no drugs:0,inhibitors:1,namenda:2,hypertension:3,inhibitors+namenda:4,others:5}



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
    # elif '-4' in med_list:
    #     action=0
    else:
        action=0

    action_list.append(action)
merged_data2= merged_data.assign(action = action_list)
print(merged_data2['action'].value_counts())

merged_data25=merged_data2


#printing the dimension of data and all column names

print("\ndata dimension= ", merged_data25.shape)
print("\ncolumn_names= ", merged_data25.columns)

##demographics

#gender

male_female=merged_data25.groupby(['PTGENDER'])['RID'].nunique()
print("\n", male_female)

###total patients

total_patients=len(pd.unique(merged_data25['RID']))
print("\nTotal Patients= ",total_patients)

#total visits

total_visits=merged_data25['RID'].value_counts()
print("\nMean Total Visits= ", total_visits.mean())
print("\nSD Total Visits= ", total_visits.std())



#monthly visits

monthly_visits=merged_data25.groupby(['RID'])['Month'].max()
print("\nMean Monthly Visits= ", monthly_visits.mean())
print("\nSD Monthly Visits= ", monthly_visits.std())


merged_data25.to_csv('whole-data-with-states.csv', index = False)  # Export merged pandas DataFrame

