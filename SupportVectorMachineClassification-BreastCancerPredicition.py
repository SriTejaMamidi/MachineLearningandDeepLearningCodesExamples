#Importing the required modules
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,roc_auc_score
from sklearn.svm import SVC

#Creating the dataset
data=load_breast_cancer()
x=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target,name="target")
df=pd.concat([x,y],axis=1)
#Printing head and info and checking for null values
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())
x=df.drop("target",axis=1)
y=df["target"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
#Preprocessing and model pipeline
model=Pipeline(steps=[("scaler",StandardScaler()),
                      ("classifier",SVC(kernel="rbf",class_weight="balanced"))])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
roc=roc_auc_score(y_test,model.decision_function(x_test))
print("Accuracy score is",accuracy)
print("F1 score is",f1)
print("Precision score is",precision)
print("Recall score is",recall)
print("Confusion matrix is",confusion)
print("ROC AUC score is",roc)
print(df.iloc[1])
sample_data=pd.DataFrame([[
24.570000,15.770000,172.900000,1826.000000,0.094740,0.088640,0.096900,0.070170,
0.391200,0.058670,0.783500,0.933900,5.398000,74.080000,0.005225,0.013080,
0.018600,0.013400,0.013890,0.003532,24.990000,23.410000,158.800000,1956.000000,
0.123800,0.186600,0.241600,0.186000,0.275000,0.089020
]])
preds=model.predict(sample_data)
print("The target is:",preds)
