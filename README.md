# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method 

# CODING AND OUTPUT:
Skip to content
Navigation Menu
23004345
/
EXNO-4-DS

Type / to search
Code
Pull requests
Actions
Projects
Security
Insights
Owner avatar
EXNO-4-DS
Public
forked from DHINESH-SEC/EXNO-4-DS
23004345/EXNO-4-DS
Go to file
t
Add file
This branch is 1 commit ahead of DHINESH-SEC/EXNO-4-DS:main.
Folders and files
Name		
Latest commit
23004345
23004345
Update README.md
ef2830d
 · 
2 days ago
History
EXNO_4_Feature_Scaling_and_Selection.ipynb
Add files via upload
9 months ago
README.md
Update README.md
2 days ago
bmi.csv
Add files via upload
9 months ago
exno_4_feature_scaling_and_selection.py
Add files via upload
9 months ago
titanic_dataset.csv
Add files via upload
9 months ago
Repository files navigation
README
EXNO:4-DS
AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file.

ALGORITHM:
STEP 1:Read the given Data. STEP 2:Clean the Data Set using Data Cleaning Process. STEP 3:Apply Feature Scaling for the feature in the data set. STEP 4:Apply Feature Selection for the feature in the data set. STEP 5:Save the data to the file.

FEATURE SCALING:
Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).
FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well. The feature selection techniques used are: 1.Filter Method 2.Wrapper Method 3.Embedded Method

CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/9e7b4e0d-b1e1-42b0-8226-2685576aea15)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/a2a4526f-bea9-4592-9a04-1e422e23b510)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/d9353132-b13c-4dd1-85cf-db0fbb431dc1)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/a124bb94-e580-4cb4-923d-f89bd7b3e932)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/e6ccebff-f2ef-40e3-9409-f82e0c1da7c8)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/3fbeb0f3-c491-42a8-90cc-6f3c4268554f)
```
data2
```
![image](https://github.com/user-attachments/assets/a7ec9cba-c72c-4c41-a985-46ba5e94edfb)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/a43b8469-67e1-472d-ae12-8606b82d7ace)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/2a0941a4-cef2-45c8-88c1-2539d890fb73)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/82796029-a89f-4b50-b233-e7dde57a1ea4)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/e2fbfa36-0bd3-4c9a-bcf3-766ba57efb9c)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/ae3ba7f5-0299-4702-8db9-59e113f26d0f)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/9a7596cc-d3a0-4308-b687-3edeebbfc168)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/ed0eb7fd-f6a6-49e5-be50-41d7025246a5)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/1720a0a0-0c40-40af-accd-5c94d1fc5dfa)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/213dc273-4d70-4e03-911b-2a0a899a7671)
```
data.shape
```
![image](https://github.com/user-attachments/assets/ed3e3022-1637-48af-b299-8ec4fe23cab5)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/7ddab293-4b2c-486a-994d-794d086124fe)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/8ce7d581-8138-4c26-8861-be04fe69198e)
```
tips.time.unique()
```
![image](https://github.com/user-attachments/assets/842de1a9-2982-4adb-8f8c-1f68a0bd8504)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/eedeebc9-7010-436d-a85a-a3bc9675e497)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/b53afffb-24e8-4466-acf9-ca70cae0ae95)

# RESULT:
      Thus, Feature selection and Feature scaling has been used on thegiven dataset.
