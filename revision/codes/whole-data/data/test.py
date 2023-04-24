import pandas as pd
import numpy as np
from ast import literal_eval

pd.set_option('display.max_rows', 500)

merged_file= pd.read_csv("/Users/kritibbhattarai/Desktop/Treatment_optimization_AD/revision/codes/whole-data/data/whole-data-without-states.csv")

merged_file['CMMED']=merged_file['CMMED'].apply(lambda x:"['-4']" if pd.isna(x) else x)
# merged_file['CMMED']=merged_file['CMMED'].apply(lambda x:"['-4']" if np.nan in x else x)
merged_file['CMMED'].isnull().values.any()
# merged_file["CMMED"] = merged_file["CMMED"].fillna(["-4"])
merged_file['CMMED'] = merged_file['CMMED'].apply(literal_eval)
contains_non_string = merged_file[merged_file['CMMED'].apply(type) != str].any()
print(contains_non_string)
merged_file.to_csv('test.csv', index = False)  # Export merged pandas DataFrame
print(type(merged_file.iloc[2]['CMMED']))
