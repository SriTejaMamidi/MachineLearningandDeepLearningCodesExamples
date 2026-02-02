#Importing the required modules
import numpy as np
import pandas as pd
from sklearn.svm import SVR, SVC
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score

#Creating dataset
data=fetch_california_housing()
x=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target,name="target")
df=pd.concat([x,y],axis=1)
#Printing head and info and checking for null values
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
x=df.drop("target",axis=1)
y=df["target"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#Preprocessing and model pipeline
model=Pipeline(steps=[("scaler",StandardScaler()),
                      ("Regression",SVR(kernel="rbf",epsilon=0.1,C=100))])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
rmse=root_mean_squared_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
print("RMSE:",rmse)
print("MAE:",mae)
print("R2:",r2)
print("MSE",mse)
sample_data=pd.DataFrame([{"MedInc":9.5,
                           "HouseAge":20,
                           "AveRooms":7.1,
                           "AveBedrms":1,
                           "Population":2500,
                           "AveOccup":3,
                           "Latitude":37.860000,
                           "Longitude":-122.220000}])
prediction=model.predict(sample_data)
print("Recommendation is:",prediction)