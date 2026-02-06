#Importing the required modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN,KMeans,AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
#Getting the data
x,_=make_blobs(n_samples=500,n_features=2,centers=4,cluster_std=1.2,random_state=42)
df=pd.DataFrame(x,columns=["Annual_Income","Spending_Score"])
x=df.copy()
x=df.select_dtypes(include="number")
scaler = StandardScaler()
scaler.fit(x)
x_scaled=scaler.transform(x)
#First we will do hierarchichal
plt.figure(figsize=(10,6))
sch.dendrogram(sch.linkage(x_scaled,method="ward"))
plt.title("Dendrogram")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()
#Final clustering
hc=AgglomerativeClustering(n_clusters=4,linkage='ward')
labels=hc.fit_predict(x_scaled)
df["Cluster"]=labels
print(df["Cluster"].value_counts())
#Now we will do DBSCAN
dbscan=DBSCAN(eps=0.2,min_samples=8)
labels=dbscan.fit_predict(x_scaled)
df["Cluster"]=labels
noisepoints=df[df["Cluster"]==-1]
#Visualization
print("Noise points:",len(noisepoints))
plt.scatter(x_scaled[:,0],x_scaled[:,1],c=labels)
plt.title("DBSCAN Clustering")
plt.show()
df["Cluster"]=labels
print(df["Cluster"].value_counts())
#Now we will do kmeans
inertia=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit(x_scaled)
    inertia.append(kmeans.inertia_)
kmeans=KMeans(n_clusters=4,random_state=42)
clusters=kmeans.fit_predict(x_scaled)
df["Cluster"]=clusters
for k in range(2,10):
    label=KMeans(n_clusters=k,random_state=42).fit_predict(x_scaled)
    score2=silhouette_score(x_scaled,label)
    print(f"k={k},silhouette_score={score2:3f}")

score=silhouette_score(x_scaled,df["Cluster"])
print("Silhouette Score:",score)
#Now will predict
new_customers=[[60000,75]]
new_customers_scaled=scaler.transform(new_customers)
cluster_id=kmeans.predict(new_customers_scaled)
print("Assigned cluster id:",cluster_id)