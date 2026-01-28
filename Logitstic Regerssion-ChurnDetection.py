#Importing the required modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,
                             precision_score, recall_score, roc_auc_score, classification_report,roc_curve,confusion_matrix)
import matplotlib.pyplot as plt
#Loading the dataset
df=pd.read_csv('customer_churn.csv')
#Printing the head and info and checking for null values
print(df.head())
print(df.info())
#Churn gives the result of 0 and 1
print(df['Churn'].value_counts())
#Giving the input and outputs
x=df.drop(columns=["CustomerID","Churn"])
y=df['Churn']
#Converting categorical into numerical
cat_cols=[
    "Gender",
    "ContractType",
    "PaymentMethod",
    "InternetService"
]
num_cols=[
    "Age",
    "TenureMonths",
    "MonthlyCharges",
    "TotalCharges",
    "SupportCalls"
]
#Making the preprocessing and pipeline
preprocessor=ColumnTransformer(
    transformers=[
    ("cat",OneHotEncoder(drop="first"),cat_cols),
    ("num",StandardScaler(),num_cols)
])
model=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',LogisticRegression(class_weight='balanced'))
    ])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=48,stratify=y)
model.fit(x_train,y_train)
#Making the predictions now
y_pred=model.predict(x_test)
y_pred_prob=model.predict_proba(x_test)[:,1]
#Evaluation metrics
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("ROC_AUC:",roc_auc_score(y_test,y_pred_prob))
print("\nClassification Report")
print(classification_report(y_test,y_pred))
#Threshold tuning
custom_threshold=0.4
y_custom=(y_test>custom_threshold).astype(int)
print("Precision(0.4):",precision_score(y_test,y_custom))
print("Recall(0.4):",recall_score(y_test,y_custom))
print("Confusion matrix:",confusion_matrix(y_test,y_custom))
train_pred=model.predict(x_train)
print("Train accuracy:",accuracy_score(y_train,train_pred))
print("Test accuracy:",accuracy_score(y_test,y_pred))
#ROC curve
fpr,ftr,threshold=roc_curve(y_test,y_pred_prob)
plt.figure()
plt.plot(fpr,ftr)
plt.plot([0,1],[0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

