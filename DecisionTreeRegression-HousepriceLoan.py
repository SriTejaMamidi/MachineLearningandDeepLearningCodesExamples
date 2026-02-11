#Importing the required modules
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,r2_score

# Set seed for reproducibility
np.random.seed(42)

n_rows = 20000


# ----- House Prices Dataset (Regression) -----
house_data = pd.DataFrame({
    'House_ID': ['HS'+str(i).zfill(5) for i in range(1, n_rows+1)],
    'Bedrooms': np.random.choice([1,2,3,4,5], n_rows, p=[0.1,0.2,0.4,0.2,0.1]),
    'Bathrooms': np.random.choice([1,2,3], n_rows, p=[0.3,0.5,0.2]),
    'Sqft_Living': np.random.normal(1500, 500, n_rows).astype(int),
    'Sqft_Lot': np.random.normal(5000, 2000, n_rows).astype(int),
    'Floors': np.random.choice([1,2,3], n_rows, p=[0.5,0.4,0.1]),
    'Waterfront': np.random.choice([0,1], n_rows, p=[0.95,0.05]),
    'View': np.random.choice([0,1,2,3,4], n_rows, p=[0.5,0.2,0.15,0.1,0.05]),
    'Condition': np.random.choice([1,2,3,4,5], n_rows, p=[0.05,0.15,0.6,0.15,0.05]),
    'Grade': np.random.choice(range(1,11), n_rows),
    'Sqft_Above': np.random.normal(1200, 400, n_rows).astype(int),
    'Sqft_Basement': np.random.normal(300, 200, n_rows).astype(int),
})

house_data['Price'] = (
    house_data['Sqft_Living']*300 +
    house_data['Bedrooms']*5000 +
    house_data['Bathrooms']*3000 +
    house_data['View']*2000 +
    house_data['Grade']*10000 +
    np.random.normal(0,50000,n_rows)
).astype(int)

house_data.to_csv('house_prices_dataset.csv', index=False)
print("House Prices CSV created!")
df=pd.read_csv("house_prices_dataset.csv")
#Printing head and info for null values
print(df.head())
print(df.shape)
print(df.describe())
print(df.isnull().sum())
print(df.info())
#Feature Engineer
x=df.drop(["Price"],axis=1)
y=df["Price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#Preprocessing and pipeline
preprocess=ColumnTransformer([
    ('cat',OneHotEncoder(handle_unknown='ignore'),['House_ID'])
])
model=Pipeline(steps=[
    ('preprocessor',preprocess),
    ('drt',DecisionTreeRegressor(criterion="squared_error",
                          max_depth=6,
                          min_samples_leaf=5,
                          min_samples_split=10,
                          random_state=42))
])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
rm=root_mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r=r2_score(y_test,y_pred)
print("R2 score is",r)
print("MAE is",mae)
print("MSE is",mse)
print("RMSE is",rm)
#Prediction
samples=[
['HS00001',4,3,1400,5098,3,1,2,4,7,777,900],
['HS00003',4,3,1700,5898,3,1,3,4,8,877,1000]
]
sample_df = pd.DataFrame(samples, columns=x.columns)
sample_pred = model.predict(sample_df)
for i, Price in enumerate(sample_pred):
    print(f"Sample Loan amount{i + 1} Predicted price is {Price:.2f}")
