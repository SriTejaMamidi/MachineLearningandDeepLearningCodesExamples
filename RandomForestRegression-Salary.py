#Importing the modules
import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score

from LR import predicted_salary

df=pd.read_csv('Salary_Data[1].csv')
#Chekcing the head and null values
df.columns=df.columns.str.strip()
print(df.head())
print(df.info())
#Filling the null values
df['Salary']=df['Salary'].fillna(df['Salary'].median())
num_cols=['Age','Years of Experience']
df[num_cols]=df[num_cols].fillna(df[num_cols].median())
cat_cols=['Gender','Education Level','Job Title']
df[cat_cols]=df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
#Checking again if there are any null values
print(df.info())
#Giving input and outputs
x=df.drop('Salary',axis=1)
y=df['Salary']
preprocessor=ColumnTransformer(
    transformers=[
    ('cat',OneHotEncoder(handle_unknown='ignore'),cat_cols),
    ('num','passthrough',num_cols)
])
model=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('rf',RandomForestRegressor(n_estimators=200,random_state=42))
    ])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=48)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))
#predicition
new_data=pd.DataFrame({
    'Age':[35],
    'Years of Experience':[8],
    'Gender':['Male'],
    'Education Level':['Bachelors'],
    'Job Title':['Software Engineer']
})
predicted_salary=model.predict(new_data)
print(predicted_salary)