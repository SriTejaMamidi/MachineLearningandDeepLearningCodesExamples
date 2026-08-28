#Importing the modules required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score)
#Reading the file and printing the head and info
df=pd.read_csv("customer_churn_large.csv")
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df.info())
#As internet service column is not full we are filling it
df["InternetService"]=df["InternetService"].fillna(df["InternetService"].mode()[0])
#Rechecking it
print(df.info())
#Featuring the input and output
x=df.drop(["CustomerID","Churn"],axis=1)
y=df["Churn"]
#Converting the categorical into numerical columns
cat_cols=["Gender","ContractType","PaymentMethod","InternetService"]
num_cols=["Age","TenureMonths","MonthlyCharges","SupportCalls","TotalCharges"]
#Preprocessing and the pipeline
preprocessing=ColumnTransformer(
    transformers=[
        ("cat",OneHotEncoder(drop="first"),cat_cols),
        ("num",StandardScaler(),num_cols),
    ]
)
model=Pipeline([
    ("preprocessing",preprocessing),
    ("classifier",LogisticRegression(class_weight='balanced')),
])
#Training the model and fitting it
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=48,stratify=y)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred_prob=model.predict_proba(x_test)[:,1]
#Evaluating metrics
print("Accuracy score",accuracy_score(y_test,y_pred))
print("Precision score",precision_score(y_test,y_pred))
print("Recall score",recall_score(y_test,y_pred))
print("F1 score",f1_score(y_test,y_pred))
print("classification report",classification_report(y_test,y_pred))
print("confusion matrix",confusion_matrix(y_test,y_pred))
print("ROC AUC score",roc_auc_score(y_test,y_pred))
train_pred=model.predict(x_train)
print("Train accuracy:",accuracy_score(y_train,train_pred))
print("Test accuracy:",accuracy_score(y_test,y_pred))
