#Importing the required modules
import numpy as np
import pandas as pd
from keras.src.layers import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,precision_score,recall_score,classification_report
#Reading the file and checking for null values
df=pd.read_csv("credit_risk_combined_dataset.csv")
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())
#Giving inputs and outputs
x=df.drop(["default"],axis=1)
y=df["default"]
cat_cols=x.select_dtypes(exclude="number").columns
num_cols=x.select_dtypes(include="number").columns
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
preprocess=ColumnTransformer(transformers=[
    ("num",SimpleImputer(strategy="median"),num_cols),
    ("cat",Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ]),cat_cols)
])
#Training the model and pipeline
model=Pipeline(steps=[
    ("pre",preprocess),
    ("mod",RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced"))
])
model.fit(x_train,y_train)
y_train_pred=model.predict(x_train)
y_pred=model.predict(x_test)
#Training metrics are printed to check overfit or underfit
print("Accuracy score  for training is",accuracy_score(y_train,y_train_pred))
print("F1 score  for training is",f1_score(y_train,y_train_pred))
print("precision score  for training is",precision_score(y_train,y_train_pred))
print("Recall score  for training is",recall_score(y_train,y_train_pred))
print("Test data results")
#Test metrics represent the real world model perfomance
print("Accuracy score is",accuracy_score(y_test,y_pred))
print("F1 score is",f1_score(y_test,y_pred))
print("Precision score is",precision_score(y_test,y_pred))
print("Recall score is",recall_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
print("Confusion Matrix is",confusion_matrix(y_test,y_pred))
samples=pd.DataFrame(
    [[59,633236,38,770799,406,48,"Graduate","Salaried","Single",12.092243441456585],
[49,763236,38,770799,987,48,"Graduate","Salaried","Married",10.092243441456585]
     ],columns=x.columns
)
prd=model.predict(samples)
print("Predictied Rate is:",prd)
print(df["default"].describe())
