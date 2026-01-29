#Importing the modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,root_mean_squared_error
#Reading the document and checking for null values
df=pd.read_csv("knn_productrecommendation_regression.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
#Feature engineering
x=df.iloc[:,:-1]
y=df["Rating"]
#To store results we will create a dataset results
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
results=[]
#To find the k we have used for loop and gor the k value as 5
for k in range(1,11):
    model=Pipeline([('scaler',StandardScaler()),
                    ('knn',KNeighborsRegressor(n_neighbors=k))
                     ])
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    msee=mean_squared_error(y_test,y_pred)
    rt=root_mean_squared_error(y_test,y_pred)
    results.append((k,msee,rt))

for k,msee,rt in results:
    print(k,msee,rt)

results_df=pd.DataFrame(results,columns=['k','msee','rt'])
best_k=results_df.loc[results_df["rt"].idxmin(),"k"]
#Final pipeline
final_model=Pipeline([('scaler',StandardScaler()),
                      ('knn',KNeighborsRegressor(n_neighbors=best_k))
                       ])
final_model.fit(x_train,y_train)
y_pred_final=final_model.predict(x_test)
final_msee=mean_squared_error(y_test,y_pred_final)
final_rt=root_mean_squared_error(y_test,y_pred_final)
print("\nFinal model perfomance:")
print("Final MSE:",final_msee,"\n")
print("Final RMSE:",final_rt,"\n")
#Now we will predict using the model
sample_data=[[30,12.0,40,6,0.7]]
prediction=final_model.predict(sample_data)
print("Rating is:",prediction)