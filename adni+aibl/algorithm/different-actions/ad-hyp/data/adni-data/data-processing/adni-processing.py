import numpy as np
import pandas as pd

data_path='/Users/kritibbhattarai/Desktop/Treatment_optimization_AD/adni+aibl/adni-data/raw-data/'
adni_merge= pd.read_csv(data_path+'ADNIMERGE_May15.2014.csv')
cdr=pd.read_csv(data_path+'CDR.csv')
labdata=pd.read_csv(data_path+'LABDATA.csv')
neurobat=pd.read_csv(data_path+'NEUROBAT.csv')

selected_adni_merge=adni_merge[['RID','EXAMDATE', 'MMSE']]
selected_adni_merge['EXAMDATE'] = pd.to_datetime(selected_adni_merge['EXAMDATE']).dt.strftime('%y-%m')

cdr['EXAMDATE'] = pd.to_datetime(cdr['EXAMDATE']).dt.strftime('%y-%m')

labdata['EXAMDATE'] = pd.to_datetime(labdata['EXAMDATE']).dt.strftime('%y-%m')

neurobat['EXAMDATE'] = pd.to_datetime(neurobat['EXAMDATE']).dt.strftime('%y-%m')
# print(selected_adni_merge.head(50))

merge_1=pd.merge(selected_adni_merge, cdr[['RID','EXAMDATE','CDGLOBAL']], on=['RID', 'EXAMDATE'], how='left')
merge_2=pd.merge(merge_1, labdata[['RID','EXAMDATE','AXT117','BAT126','HMT3','HMT7','HMT13','HMT40','HMT100','HMT102']], on=['RID','EXAMDATE'], how='left')

merge_3=pd.merge(merge_2, neurobat[['RID','EXAMDATE','LIMMTOTAL','LDELTOTAL']], on=['RID','EXAMDATE'], how='left')

lab_file=merge_3

lab_file=lab_file.groupby(['RID']).apply(lambda x: x.ffill().bfill())
lab_file=lab_file.groupby(['RID']).apply(lambda x: x.fillna(method='bfill').fillna(method='ffill'))

cols=['AXT117','BAT126','HMT3','HMT7','HMT13','HMT40','HMT100','HMT102','CDGLOBAL', 'LIMMTOTAL','LDELTOTAL']


for col in cols:
    lab_file[col] = pd.to_numeric(lab_file[col], errors='coerce')
    lab_file[col] = lab_file[col].fillna((lab_file[col].mean()))


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

##medication file
medication_file=pd.read_csv("/Users/kritibbhattarai/Downloads/AD_main/RECCMEDS.csv")

##converting CMMED and CMREASON to lowercase

medication_file["CMMED"]=medication_file["CMMED"].str.lower()
medication_file["CMREASON"]=medication_file["CMREASON"].str.lower()

##replacing strings in CMMED
replace_MED={
        "nepezil":"aricept",
        "aricept": "aricept",
        "memantine":"namenda",
        "mementine":"namenda",
        "menda": "namenda",
        "zadyne": "razadyne",
        "galantamine":"razadyne",
        "rivastigmine":"exelon",
        "exelon":"exelon",
        }
    
for key, value in replace_MED.items():
    medication_file.loc[medication_file['CMMED'].str.contains(key,na=False), 'CMMED'] = value


##replacing strings in CMREASON

replace_REASON={
        "alzheimer":"ad",
        "sleep":"sleep",
        "insomnia":"sleep",
        "ad":"ad",
        "vitamin":"vitamin",
        "memory":"ad",
        "acid":"acidity",
        "depress":"depression",
        "dementia":"ad",
        "congitive":"ad",
        "mild cognitive impairment":"ad",
        "mci":"ad",
        "cogni": "ad",
        "aller":"allergy",
        "aspirin":"pain",
        "pain":"pain",
        "pressure":"blood pressure",
        "cholesterol":"cholesterol",
        "supplement":"supplement",
        }


for key, value in replace_REASON.items():
    medication_file.loc[medication_file['CMREASON'].str.contains(key,na=False), 'CMREASON'] = value



#giving all the inhibitors the same name

inhibitors_list=["aricept","razadyne", 'exelon', "Tacrine",	"Pyridostigmine", "Galantamine"	,"Isoflurophate", "Donepezil" ,"Demecarium","Physostigmine"	,"Rivastigmine","Edrophonium"	,"Ambenonium","Neostigmine"	,"Distigmine","Malathion","Echothiophate",	"1,10-Phenanthroline"	,"Huperzine A"	,"Coumaphos","Dichlorvos","Fenthion","Metrifonate","Methanesulfonyl Fluoride"	,"Paraoxon", "Ginkgo biloba", "Ipidacrine","Gallamine triethiodide", "Tubocurarine", "Decamethonium", "Minaprine", "Mefloquine	A", "Metoclopramide"	,"Tyrothricin","Aprotinin"	,"Thiotepa","Mechlorethamine","Tretamine","Hexafluronium","Profenamine","Chlorpromazine","Cinchocaine","Terbutaline","Procainamide","Pancuronium","Bambuterol","Phenserine","Betaine","Capsaicin","Diethylcarbamazine","Nizatidine","Pegvisomant","Perindopril","Pipecuronium","Cisplatin","Procaine","Ramipril","Regramostim","Sulpiride","Triflupromazine","Ketamine","Ranitidine","Acotiamide","Posiphen","Methylphosphinic Acid"	,"Carbaryl"]
[item.lower() for item in inhibitors_list]

for medicine in inhibitors_list:
    medication_file['CMMED'] = medication_file['CMMED'].str.replace(medicine,'inhibitors')

medication_file['CMMED'] = np.where((medication_file['CMREASON'] == "alzheimer's") & (medication_file['CMMED'] != 'namenda'), 'inhibitors', medication_file['CMMED'])



###giving all hypertension drugs the same name

hypertension_list=['Acebutolol', 'Aliskiren', 'Amiloride', 'Amlodipine', 'Ammoniumchloride', 'Atenolol', 'Azilsartanmedoxomil', 'Benazepril', 'Bendroflumethiazide', 'Benidipine', 'Betaxolol', 'Bisoprolol', 'Candesartan', 'Candesartancilexetil', 'Canrenoicacid', 'Captopril', 'Carvedilol', 'Celiprolol', 'Chlorothiazide', 'Chlorthalidone', 'Cilazapril', 'Cilnidipine', 'Clevidipine', 'Clonidine', 'Clopamide', 'Delapril', 'Diltiazem', 'Doxazosin', 'Enalapril', 'Enalaprilat', 'Eplerenone', 'Eprosartan', 'Esmolol', 'Felodipine', 'Fosinopril', 'Furosemide', 'Guanfacine', 'Hydrochlorothiazide', 'Hydroflumethiazide', 'Indapamide', 'Irbesartan', 'Isradipine', 'Labetalol', 'Lacidipine', 'Lercanidipine', 'Levamlodipine', 'Lisinopril', 'Losartan', 'Methyclothiazide', 'Methyldopa', 'Metolazone', 'Metoprolol', 'Moexipril', 'Moxonidine', 'Nadolol', 'Nebivolol', 'Nicardipine', 'Nifedipine', 'Nisoldipine', 'Nitrendipine', 'Olmesartan', 'Perindopril', 'Pindolol', 'Piretanide', 'Polythiazide', 'Prazosin', 'Propranolol', 'Quinapril', 'Ramipril', 'Reserpine', 'Rilmenidine', 'Rosuvastatin', 'Spironolactone', 'Telmisartan', 'Terazosin', 'Topiramate', 'Torasemide', 'Trandolapril', 'Triamterene', 'Urapidil', 'Valsartan', 'Verapamil', 'Xipamide', 'Zofenopril']
[item.lower() for item in hypertension_list]

for medicine in hypertension_list:
    medication_file['CMMED'] = medication_file['CMMED'].str.replace(medicine,'hypertension')

medication_file.loc[medication_file.CMREASON=="hypertension", 'CMMED']='hypertension'


main_medicines=['inhibitors', 'namenda', 'hypertension']

###replacing date column names


updated_file=medication_file.rename(columns={'EXAMDATE':'EXAMDATE_MEDS', 'USERDATE':"EXAMDATE"},inplace=False)
updated_file['EXAMDATE'] = pd.to_datetime(updated_file['EXAMDATE']).dt.strftime('%y-%m')


###merging medication rows for same visit day

# updated_file=updated_file[~updated_file['VISCODE'].str.contains("bl", na=False)]
updated_file=updated_file.groupby(['RID','EXAMDATE'], as_index=False).agg({'CMMED': list, 'CMREASON':list}).reset_index()
# print("updated", updated_file.head(50))
updated_file=updated_file.sort_values("RID")

# updated_file.to_csv('med.csv',date_format='%y-%m', index=False)

# print(updated_file.head(10))

merged_data= pd.merge(lab_file, updated_file, on=['RID', 'EXAMDATE'], how='left')

merged_data=merged_data.groupby(['RID']).apply(lambda x: x.ffill().bfill())


merged_data=merged_data.dropna(subset=['CMREASON'])
def check_group(group):
    if isinstance(group['CMREASON'], pd.Series) or isinstance(group['CMMED'], pd.Series):
        for row_id, row in group.iterrows():
            if set(['hypertension','ad']).intersection(set(row['CMREASON'])) or set(['hypertension','inhibitors','namenda']).intersection(set(row['CMMED'])): 
                return group
merged_data1= merged_data.groupby('RID').apply(check_group)

merged_data1.to_csv('processed_ad_hyp_lab_file.csv',date_format='%y-%m', index=False)

