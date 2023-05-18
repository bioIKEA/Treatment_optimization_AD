import pandas as pd
import numpy as np
from ast import literal_eval

pd.set_option('display.max_rows', 500)

data_path='/Users/kritibbhattarai/Desktop/Treatment_optimization_AD/adni+aibl/algorithm/ad-hyp/data/adni-data/data-processing/'


merged_file= pd.read_csv("processed_ad_hyp_lab_file.csv")


############assigining_states####################

state=0
state_list=[]
for index,row in merged_file.iterrows():
    if ( row['CDGLOBAL_pre'] <= 0.75 ) :
        if ( row['LDELTOTAL_pre'] <= 3.5 ) :
            if ( row['HMT100_pre'] <= 27.049872398376465 ) :
                state=0
            else :
                if ( row['LIMMTOTAL_pre'] <= 2.5 ) :
                    state=1
                
                else :
                    state=2
        else :
            if ( row['LIMMTOTAL_pre'] <= 10.5 ) :
                if ( row['BAT126_pre'] <= 532.5 ) :
                    state=3
                
                else :
                    state=4
            
            else :
                state=5
    else :
        if ( row['LIMMTOTAL_pre'] <= 4.5 ) :
            if ( row['LIMMTOTAL_pre'] <= 1.5 ) :
                state=6
            else :
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

merged_data25.to_csv('adni_ad_hyp_lab_w_states.csv', index = False)  # Export merged pandas DataFrame



