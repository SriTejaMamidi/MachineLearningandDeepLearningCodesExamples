#Importing the required modules
import numpy as np
import pandas as pd
#Importing data into df
df=pd.read_csv('Salary_Data[1].csv')
#Checking head and null values
print(df.head())
print(df.info())
#Filling the null values
df['Education Level'].fillna(df['Education Level'].mode()[0],inplace=True)
df['Job Title'].fillna(df['Job Title'].mode()[0],inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Years of Experience'].fillna(df['Years of Experience'].mode()[0],inplace=True)
df['Salary'].fillna(df['Salary'].median(),inplace=True)
df['Age'].fillna(df['Age'].median(),inplace=True)
#Rechecking if there are null values
print(df.info())
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
#Encoding categorical values
df_encoded=pd.get_dummies(df,columns=['Job Title','Gender','Education Level'],drop_first=True)
#Featuring inputs and output
x=df_encoded.drop('Salary',axis=1)
y=df_encoded['Salary']
#Train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=48)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
#Linear Regression
model=LinearRegression()
model.fit(x_train_scaled,y_train)
#Evalulating the model
y_pred=model.predict(x_test_scaled)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(mse,r2)
#Predict new salary
new_experience_prediction=np.array([[12,35]])
new_experience_scaled=scaler.transform(new_experience_prediction)
prediction=model.predict(new_experience_scaled)
print("\nPredicted salary is",prediction)

