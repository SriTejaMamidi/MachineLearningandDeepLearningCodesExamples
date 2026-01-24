import pandas as pd
df=pd.read_csv('employee_attrition_dirty.csv')
#Checking if there are any null values in the data
print(df.head())
print(df.info())
#Filling the null values using mean and median
df['salary'].fillna(df['salary'].median(), inplace=True)
df['education'].fillna(df['education'].mode()[0], inplace=True)
df['department'].fillna(df['department'].mode()[0], inplace=True)
df['gender'].fillna(df['gender'].mode()[0], inplace=True)
#Checking again if there are null values
print(df.info())
#Now removing the duplicate files
df.drop_duplicates(inplace=True)
#Changing categorical values
df_encoded=pd.get_dummies(df,columns=['salary','education','department','gender'],drop_first=True)
#Now scaling and Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_encoded[['age','salary','years_at_company']]=scaler.fit_transform(df_encoded[['age','salary','years_at_company']])
#Now feature Engineering
df_encoded['salary_per_year']=df['salary']/(df['years_at_company']+1)
df_encoded['is_senior']=(df['years_at_company']>5).astype(int)
