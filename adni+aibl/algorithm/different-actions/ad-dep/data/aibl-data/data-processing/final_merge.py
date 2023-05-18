import pandas as pd
import numpy as np
from ast import literal_eval

pd.set_option('display.max_rows', 500)

data_path='/Users/kritibbhattarai/Desktop/Treatment_optimization_AD/adni+aibl/aibl-data/data-processing/'

merged_file= pd.read_csv("processed_aibl_lab_file.csv")


############assigining_states####################

state=0
state_list=[]
for index,row in merged_file.iterrows():
    if ( row['CDGLOBAL_pre'] <= 0.75 ) :
        if ( row['LDELTOTAL_pre'] <= 7.75138258934021 ) :
            if ( row['CDGLOBAL_pre'] <= 0.4630177468061447 ) :
                if ( row['BAT126_pre'] <= 466.0 ) :
                    state=0
                
                else :
                    state=1
                
            
            else :
                if ( row['LIMMTOTAL_pre'] <= 5.5 ) :
                    state=2
                
                else :
                    state=3
        
        else :
            if ( row['LDELTOTAL_pre'] <= 9.5 ) :
                state=4
            else :
                state=5
    else :
        if ( row['LIMMTOTAL_pre'] <= 3.5 ) :
            state=6
        
        else :
            if ( row['LIMMTOTAL_pre'] <= 6.5 ) :
                state=7
                
            
            else :
                state=8
                
    state_list.append(state)
merged_file = merged_file.assign(states= state_list)

merged_file.to_csv('aibl_lab_w_states.csv', index = False)  # Export merged pandas DataFrame



