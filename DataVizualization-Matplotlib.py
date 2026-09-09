#Importing the modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('business_kpi_dashboard.csv')
print(df.head())
print(df.info())
df['Date']=pd.to_datetime(df['Date'])
df['Net_Profit']=df['Profit']-df['Loss']
df['Profit_Margin']=df['Net_Profit']/df['Sales']
Total_Sales=df['Sales'].sum()
Total_Profit=df['Profit'].sum()
Total_loss=df['Loss'].sum()
net_profit=df['Net_Profit'].sum()
avg_profit_mean=df['Profit_Margin'].mean()
#Distribution plots
plt.figure()
sns.histplot(df['Sales'],kde=True)
plt.title('Sales Distribution')
plt.show()
plt.figure()
sns.histplot(df['Profit'],kde=True)
plt.title('Profit Distribution')
plt.show()
plt.figure()
sns.boxplot(df['Units_Sold'])
plt.title('Units Sold Distribution')
plt.show()
#Correlation Heatmap
plt.figure()
corr=df[['Sales','Profit','Loss','Units_Sold']].corr()
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
#Time Series Analysis
daily_sales=df.groupby('Date')['Sales'].sum()
plt.figure()
daily_sales.plot()
plt.title('Daily Sales Distribution')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()
#Profit and Loss
plt.figure()
sns.scatterplot(x='Sales',y='Net_Profit').sum()
plt.title('Net Profit Distribution')
plt.show()
