#Importing the required modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error
#Giving data to df and checking for null values
df=pd.read_csv("credit_risk_combined_dataset.csv")
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df.info())
#Giving inputs and outputs
x=df.drop("interest_rate",axis=1)
y=df["interest_rate"]
cat_cols=x.select_dtypes(exclude="number").columns
num_cols=x.select_dtypes(include="number").columns
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
preprocess=ColumnTransformer(transformers=[
    ("num",SimpleImputer(strategy="median"),num_cols),
    ("cat",Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ]),cat_cols)
])
#Pipeline and fitting it
model=Pipeline(steps=[
    ("preprocessor",preprocess),
    ("mod",RandomForestRegressor(random_state=42,n_estimators=200))
])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_train_pred=model.predict(x_train)
print("MSE is",mean_squared_error(y_test,y_pred))
print("R2 is",r2_score(y_test,y_pred))
print("RMSE is",root_mean_squared_error(y_test,y_pred))

print("Train metrics")
mse_train=mean_squared_error(y_train,y_train_pred)
rmse_train=root_mean_squared_error(y_train,y_train_pred)
r2_train=r2_score(y_train,y_train_pred)
print(mse_train,r2_train,rmse_train)
print("Test metrics")
mse_test=mean_squared_error(y_test,y_pred)
rmse_test=root_mean_squared_error(y_test,y_pred)
r2_test=r2_score(y_test,y_pred)
print(mse_test,r2_test,rmse_test)
samples=pd.DataFrame(
    [[59,633236,38,770799,406,48,"Graduate","Salaried","Single",1]],columns=x.columns
)
prd=model.predict(samples)
print("Predictied Rate is:",prd)
print(df["interest_rate"].describe())