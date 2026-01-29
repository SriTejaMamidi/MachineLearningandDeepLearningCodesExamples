#Importing the modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import LabelEncoder
#Reading the document and checking for the null values
df=pd.read_csv("knnproductrecommendation_classification.csv")
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())
#Feature engineering
x=df.iloc[:,:-1]
y=df["Recommended"]
le=LabelEncoder()
y=le.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#To store results we will create a dataset results
results=[]
for k in range(1,11):
    model=Pipeline([('scaler',StandardScaler()),
                    ('knn',KNeighborsClassifier(n_neighbors=k))
                    ])
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    f = f1_score(y_test, y_pred)
    results.append((k, acc, f))

for k,acc,f in results:
    print(k,acc,f)

results_df = pd.DataFrame(results, columns=['k', 'acc', 'f'])
best_k = results_df.loc[results_df["f"].idxmax(), "k"]
print("The best_k is:",best_k)
#Final pipeline
final_model=Pipeline([('scaler',StandardScaler()),
                      ('knn',KNeighborsClassifier(n_neighbors=best_k))
                       ])
final_model.fit(x_train,y_train)
y_pred_final=final_model.predict(x_test)
final_acc=accuracy_score(y_test,y_pred_final)
final_f=f1_score(y_test,y_pred_final)
print("\nFinal model perfomance:")
print("Final Accuracy is:",final_acc,"\n")
print("Final f1_score is:",final_f,"\n")
#Now we will predict using the model
sample_data=[[29,6.5,60,7,9]]
prediction=final_model.predict(sample_data)
print("Recommendation is:",prediction)

