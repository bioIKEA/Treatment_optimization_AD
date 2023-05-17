import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.impute import SimpleImputer
from sklearn.tree import export_graphviz
import pydotplus
import random

##setting up display limits for rows and columns 
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)

##replacing NA with forwardfill and/or backwardfill for all columns

data_path='/Users/kritibbhattarai/Desktop/Treatment_optimization_AD/adni+aibl/algorithm/ad-hyp/data/adni-data/data-processing/'


merged_file= pd.read_csv("processed_ad_hyp_lab_file.csv")

##printing the dimension of data and all column names

print("\ndata dimension= ", merged_file.shape)

##removing dot from columns like 'RAVLT.Learning'

merged_file.columns = merged_file.columns.str.replace('.', '_')

##linear regression

model_lin = sm.OLS.from_formula("MMSE~ AXT117+BAT126+HMT3+HMT7+HMT13+HMT40+HMT100+HMT102+CDGLOBAL+LIMMTOTAL+LDELTOTAL",data=merged_file)
result_lin = model_lin.fit()
print(result_lin.summary())


###decision tree

##give range to target
mmse_range=''
mmse_range_list=[]
for mmse_value in merged_file['MMSE']:
    if mmse_value>=24:
        mmse_range="normal"
    elif mmse_value>=20 and mmse_value<24:
        mmse_range="mild"
    elif mmse_value>=13 and mmse_value<20:
        mmse_range="moderate"
    elif  mmse_value<=12:
        mmse_range="severe"
    mmse_range_list.append(mmse_range)
merged_file = merged_file.assign(mmse_score_range = mmse_range_list)


#features and target variable
features_name=['AXT117_pre','BAT126_pre','HMT3_pre','HMT7_pre','HMT13_pre','HMT40_pre','HMT100_pre','HMT102_pre','CDGLOBAL_pre', 'LIMMTOTAL_pre','LDELTOTAL_pre']

features_x= merged_file[features_name]
mmse_y = merged_file.mmse_score_range # Target

# training set and testing set
random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(features_x,mmse_y, test_size=0.2, random_state=1) 

#create imputer as suggested by the error to avoid NaN problem

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X_train)

#imputing data
X_train_imp = imp.transform(X_train)
X_test_imp=imp.transform(X_test)

#Decision Tree classifer object
d_tree= DecisionTreeClassifier(max_depth=4)

#Train Tree 
d_tree= d_tree.fit(X_train_imp,y_train)

#Predict for test data
y_prediction = d_tree.predict(X_test_imp)

##check if the model works
print("\nmodel accuracy= ",metrics.accuracy_score(y_test,y_prediction))



##display the d_tree

dot_data=tree.export_graphviz(d_tree, out_file=None,  
                filled=True, rounded=True,
                special_characters=True,feature_names =features_name ,class_names=['normal','mild', 'moderate', 'severe'])
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_png('mydecisiond_tree_second.png')


##code representation of decision tree

def get_code(tree, feature_names, target_names,
              spacer_base="    "):
     """Produce psuedo-code for decision tree.

     Args
     ----
     tree -- scikit-leant DescisionTree.
     feature_names -- list of feature names.
     target_names -- list of target (class) names.
     spacer_base -- used for spacing code (default: "    ").

     Notes
     -----
     based on http://stackoverflow.com/a/30104792.
     """
     left      = tree.tree_.children_left
     right     = tree.tree_.children_right
     threshold = tree.tree_.threshold
     features  = [feature_names[i] for i in tree.tree_.feature]
     value = tree.tree_.value

     def recurse(left, right, threshold, features, node, depth):
         spacer = spacer_base * depth
         if (threshold[node] != -2):
             print(spacer + "if ( " + features[node] + " <= " + \
                   str(threshold[node]) + " ) {")
             if left[node] != -1:
                     recurse(left, right, threshold, features,
                             left[node], depth+1)
             print(spacer + "}\n" + spacer +"else {")
             if right[node] != -1:
                     recurse(left, right, threshold, features,
                             right[node], depth+1)
             print(spacer + "}")
         else:
             target = value[node]
             for i, v in zip(np.nonzero(target)[1],
                             target[np.nonzero(target)]):
                 target_name = target_names[i]
                 target_count = int(v)
                 print(spacer + "return " + str(target_name) + \
                       " ( " + str(target_count) + " examples )")

     recurse(left, right, threshold, features, 0, 0)


print(get_code(d_tree,features_name,['normal','mild', 'moderate', 'severe']))

