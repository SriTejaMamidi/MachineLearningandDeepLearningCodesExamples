import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,root_mean_squared_error

df=pd.read_csv("Salary_Data[1].csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
preprocessor=ColumnTransformer(
    transformers=[
    ('cat',OneHotEncoder(handle_unknown='ignore'),cat_cols),
    ('num','passthrough',num_cols)
])
model=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('mod',LinearRegression())
    ])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
rmse=root_mean_squared_error(y_test,y_pred)
print("MSE:",mse)
print("R2:",r2)
print("RMSE:",rmse)
new_data=pd.DataFrame({
    'Age':[35],
    'Years of Experience':[8],
    'Gender':['Male'],
    'Education Level':['Bachelors'],
    'Job Title':['Software Engineer']
})
predicted_salary=model.predict(new_data)
print(predicted_salary)