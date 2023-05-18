import numpy as np
import pandas as pd


data_path='/Users/kritibbhattarai/Desktop/Treatment_optimization_AD/adni+aibl/aibl-data/raw-data/'

cdr=pd.read_csv(data_path+"aibl_cdr.csv")

lab_data=pd.read_csv(data_path+"aibl_labdata.csv")

neurobat=pd.read_csv(data_path+"aibl_neurobat.csv")

mmse=pd.read_csv(data_path+"aibl_mmse.csv")



merge_1=pd.merge(lab_data[['RID','VISCODE','AXT117','BAT126','HMT3','HMT7','HMT13','HMT40','HMT100','HMT102']], cdr[['RID','VISCODE','CDGLOBAL']], on=['RID', 'VISCODE'], how='left')

# merge_2=pd.merge(merge_1, lab_data[['RID','VISCODE','AXT117','BAT126','HMT3','HMT7','HMT13','HMT40','HMT100','HMT102']], on=['RID','VISCODE'], how='left')

merge_2=pd.merge(merge_1, neurobat[['RID','VISCODE','LIMMTOTAL','LDELTOTAL']], on=['RID','VISCODE'], how='left')

merge_3=pd.merge(merge_2, mmse[['RID','VISCODE','MMSCORE']], on=['RID','VISCODE'], how='left')


lab_file=merge_3
lab_file=lab_file.replace({-4:np.NaN})


lab_file=lab_file.groupby(['RID']).apply(lambda x: x.ffill().bfill())
lab_file=lab_file.groupby(['RID']).apply(lambda x: x.fillna(method='bfill').fillna(method='ffill'))

cols=['AXT117','BAT126','HMT3','HMT7','HMT13','HMT40','HMT100','HMT102','CDGLOBAL', 'LIMMTOTAL','LDELTOTAL']


for col in cols:
    lab_file[col] = pd.to_numeric(lab_file[col], errors='coerce')
    lab_file[col] = lab_file[col].fillna((lab_file[col].mean()))

# print(lab_file.head(50))

lab_file['AXT117_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['AXT117']
lab_file['BAT126_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['BAT126']
lab_file['HMT3_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['HMT3']
lab_file['HMT7_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['HMT7']
lab_file['HMT13_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['HMT13']
lab_file['HMT40_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['HMT40']
lab_file['HMT100_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['HMT100']
lab_file['HMT102_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['HMT102']
lab_file['CDGLOBAL_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['CDGLOBAL']
lab_file['LIMMTOTAL_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['LIMMTOTAL']
lab_file['LDELTOTAL_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['LDELTOTAL']


lab_file=lab_file.sort_values("RID")

# updated_file.to_csv('med.csv',date_format='%y-%m', index=False)

# print(updated_file.head(10))


lab_file.to_csv('processed_aibl_lab_file.csv', index=False)
