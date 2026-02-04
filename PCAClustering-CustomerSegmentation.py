#Importing the required modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#Getting the data
x,_=make_blobs(n_samples=500,n_features=2,centers=4,cluster_std=1.2,random_state=42)
df=pd.DataFrame(x,columns=["Annual_Income","Spending_Score"])
x=df.copy()
x=df.select_dtypes(include="number")
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
pca = PCA(n_components=2)
x_pca=pca.fit_transform(x_scaled)
print("Explained Variance:",pca.explained_variance_ratio_)
pca_df=pd.DataFrame(x_scaled,columns=["PC1","PC2"])
#Visualization
plt.scatter(pca_df["PC1"],pca_df["PC2"],color="red")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.show()