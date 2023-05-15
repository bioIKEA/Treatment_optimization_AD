import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

##setting up display limits for rows and columns 
pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 20)

##replacing NA with forwardfill and/or backwardfill for all columns

lab_file= pd.read_csv('/Users/kritibbhattarai/Downloads/AD_main/ADNIMERGE_May15.2014.csv')
lab_file=lab_file.groupby(['RID']).apply(lambda x: x.ffill().bfill())

lab_file['MMSE_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['MMSE']
lab_file['ADAS13_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['ADAS13']
lab_file['FDG_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['FDG']
lab_file['CDRSB_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['CDRSB']
lab_file['MOCA_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['MOCA']
lab_file['RAVLT.learning_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['RAVLT.learning']
lab_file['RAVLT.immediate_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['RAVLT.immediate']
lab_file['RAVLT.forgetting_pre']=lab_file.groupby(['RID']).shift(1,fill_value=0)['RAVLT.forgetting']


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
        "mementine":"nnamenda",
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


# medication_file_g= medication_file.groupby(['CMREASON'])['CMREASON'].count().reset_index(name="count_reason").sort_values(['count_reason'], ascending=False)
# print(medication_file_g)
# medication_file_g=medication_file_g[(medication_file_g['count_reason']>500)]
# print(medication_file_g)

# medication_file.plot(kind='bar', title='hi', ylabel='counts',
#          xlabel='reasons', figsize=(6, 5))
# plt.show()

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

# medication_file['CMMED'] = np.where((medication_file['CMMED']!="inhibitors" ) & (medication_file['CMMED']!="namenda" ) & (medication_file['CMMED']!="hypertension")& (medication_file['CMMED']!="-4"), "others", medication_file['CMMED'])
###replacing date column names


updated_file=medication_file.rename(columns={'EXAMDATE':'EXAMDATE_MEDS', 'USERDATE':"EXAMDATE"},inplace=False)
updated_file['EXAMDATE'] = pd.to_datetime(updated_file['EXAMDATE']).dt.strftime('%y-%m')


###merging medication rows for same visit day

# updated_file=updated_file[~updated_file['VISCODE'].str.contains("bl", na=False)]
updated_file=updated_file.groupby(['RID','EXAMDATE'], as_index=False).agg({'CMMED': list, 'CMREASON':list}).reset_index()
# print("updated", updated_file.head(50))
updated_file=updated_file.sort_values("RID")




###combining lab_file and medication_file (left join)

lab_file['EXAMDATE'] = pd.to_datetime(lab_file['EXAMDATE']).dt.strftime('%y-%m')

merged_data= pd.merge(lab_file, updated_file, on=['RID', 'EXAMDATE'], how='left')
merged_data=merged_data.groupby(['RID']).apply(lambda x: x.ffill().bfill())



# merged_data['CMREASON']=merged_data['CMREASON'].apply(lambda x:"[\"-4\"]" if pd.isna(x).any() else x)
# merged_data['CMREASON'] = merged_data['CMREASON'].apply(literal_eval)
merged_data=merged_data.dropna(subset=['CMREASON'])

def check_group(group):
        if isinstance(group['CMMED'], pd.Series):
            for row_id, row in group.iterrows():
                if 'ad' in row['CMREASON'] or 'inhibitors' in row['CMMED']or 'namenda' in row['CMMED']:
                    return group
merged_data1= merged_data.groupby('RID').apply(check_group)
print(merged_data1.shape)




merged_data1.to_csv('merged_wo-sa.csv', index = False)  # Export merged pandas DataFrame



