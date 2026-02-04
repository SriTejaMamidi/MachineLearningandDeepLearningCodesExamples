#Importing the required modules
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
#Getting the data
x,_=make_blobs(n_samples=500,n_features=2,centers=4,cluster_std=1.2,random_state=42)
df=pd.DataFrame(x,columns=["Annual_Income","Spending_Score"])
x=df.copy()
x=df.select_dtypes(include="number")
scaler = StandardScaler()
scaler.fit(x)
x_scaled=scaler.transform(x)
#Dendogram
plt.figure(figsize=(10,6))
sch.dendrogram(sch.linkage(x_scaled,method="ward"))
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
#Final clustering
hc=AgglomerativeClustering(n_clusters=4,linkage='ward')
labels=hc.fit_predict(x_scaled)
df["Cluster"]=labels
print(df["Cluster"].value_counts())