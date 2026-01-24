import numpy as np
import pandas as pd
df=pd.read_csv('sales_data.csv')
#print(df.head())
#print(df.tail())
print(df.isna().sum())
print(df.isna().any(axis=1))
print(df.info())
#Filling the missing values for region and unit price
df['region'].fillna('Unknown', inplace=True)
df['revenue'].fillna(df['revenue'].median(), inplace=True)
df['unit_price'].fillna(df['unit_price'].median(), inplace=True)
#To find the best selling product
df.groupby('product')['quantity'].sum().sort_values(ascending=False)
#To find the monthly and yearly revenue
df['order_date']=pd.to_datetime(df['order_date'])
df['year']=df['order_date'].dt.year
df['month']=df['order_date'].dt.month
monthly_revenue=df.groupby(['year','month'])['revenue'].sum()
#To find the customer retention analysis
customer_orders=df.groupby('customer_id')['order_id'].nunique()
repeat_customers=customer_orders[customer_orders>1]
print(df.info())
