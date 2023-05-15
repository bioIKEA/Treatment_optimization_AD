import pandas as pd
import numpy as np
from ast import literal_eval

pd.set_option('display.max_rows', 500)

merged_file= pd.read_csv("/Users/kritibbhattarai/Desktop/Treatment_optimization_AD/physicians_codes/whole-data/data/sr/sr_phy_wo_sa.csv")


############assigining_states####################

state=0
state_list=[]
for index,row in merged_file.iterrows():
    if ( row['ADAS13_pre'] <= 19.835000038146973 ) :
        if ( row['RAVLT.immediate_pre'] <= 30.5 ) :
            state=0
        
        else :
            state=1
        
    
    else :
        if ( row['RAVLT.immediate_pre'] <= 21.5 ) :
            state=2
        
        else :
            state=3
                
            
        
    
    state_list.append(state)
merged_file = merged_file.assign(states= state_list)







merged_data=merged_file
merged_data['CMMED']=merged_data['CMMED'].apply(lambda x:"[\"-4\"]" if pd.isna(x) else x)
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
    # if set(['inhibitors', 'namenda', 'hypertension']).issubset(set(med_list)):
    #     action=7
    # #     # print("seven",med_list)
    # elif set(['namenda', 'hypertension']).issubset(set(med_list)):
    #     action=6
        # print("six",med_list)
    # if set(['inhibitors', 'hypertension']).issubset(set(med_list)):
    #     action=5
        # print("five",med_list)
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

# print("\nmerged data dimension= ", merged_data2.shape)

merged_data25 = merged_data2[merged_data2.groupby('RID')['RID'].transform('count').ge(2)]
# print(merged_data25)


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


merged_data25.to_csv('sr_final.csv', index = False)  # Export merged pandas DataFrame



