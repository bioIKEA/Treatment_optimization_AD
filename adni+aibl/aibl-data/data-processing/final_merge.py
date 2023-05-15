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
        if ( row['LDELTOTAL_pre'] <= 4.5 ) :
            if ( row['CDGLOBAL_pre'] <= 0.21301774680614471 ) :
                
                state=0
            
            else :
                if ( row['LIMMTOTAL_pre'] <= 2.5 ) :
                    state=1
                else :
                    state=2
        
        else :
            if ( row['LDELTOTAL_pre'] <= 7.75138258934021 ) :
                if ( row['HMT3_pre'] <= 4.134697198867798 ) :
                    state=3
                
                else :
                    state=4
                
            
            else :
                if ( row['CDGLOBAL_pre'] <= 0.4630177468061447 ) :
                    state=5
                
                else :
                    state=6
    else :
        if ( row['LIMMTOTAL_pre'] <= 3.5 ) :
            if ( row['CDGLOBAL_pre'] <= 1.5 ) :
                state=7
            else :
                state=8
        else :
                state=9
                
    state_list.append(state)
merged_file = merged_file.assign(states= state_list)

merged_file.to_csv('aibl_lab_w_states.csv', index = False)  # Export merged pandas DataFrame



