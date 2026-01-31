#Importing the required modules
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
from tensorflow.python.keras.utils.generic_utils import to_list
from torch.utils.hipify.hipify_python import preprocessor

# Set seed for reproducibility
np.random.seed(42)

n_rows = 20000

# ----- Loan Approval Dataset (Classification) -----
loan_data = pd.DataFrame({
    'Loan_ID': ['LN'+str(i).zfill(5) for i in range(1, n_rows+1)],
    'Gender': np.random.choice(['Male', 'Female'], n_rows, p=[0.7, 0.3]),
    'Married': np.random.choice(['Yes', 'No'], n_rows, p=[0.6, 0.4]),
    'Dependents': np.random.choice(['0', '1', '2', '3+'], n_rows, p=[0.5, 0.2, 0.2, 0.1]),
    'Education': np.random.choice(['Graduate', 'Not Graduate'], n_rows, p=[0.8, 0.2]),
    'Self_Employed': np.random.choice(['Yes', 'No'], n_rows, p=[0.15, 0.85]),
    'ApplicantIncome': np.random.normal(5000, 2000, n_rows).astype(int),
    'CoapplicantIncome': np.random.normal(2000, 1500, n_rows).astype(int),
    'LoanAmount': np.random.normal(150, 50, n_rows).astype(int),
    'Loan_Amount_Term': np.random.choice([120, 180, 240, 300, 360, 480], n_rows, p=[0.05,0.05,0.1,0.1,0.65,0.05]),
    'Credit_History': np.random.choice([1,0], n_rows, p=[0.8,0.2]),
    'Property_Area': np.random.choice(['Urban','Semiurban','Rural'], n_rows, p=[0.3,0.4,0.3]),
})

# Simple target logic
loan_data['Loan_Status'] = np.where(
    (loan_data['Credit_History']==1) &
    (loan_data['ApplicantIncome']>4000) &
    (loan_data['LoanAmount']<200), 'Y','N'
)

loan_data.to_csv('loan_approval_dataset.csv', index=False)
print("Loan Approval CSV created!")
df=pd.read_csv('loan_approval_dataset.csv')
#Printing head and info for null values
print(df.head())
print(df.shape)
print(df.describe())
print(df.isnull().sum())
print(df.info())
#WE have written this because the data is so clean
approval_pb=(
    0.35*(df["Credit_History"]==1).astype(int)*
    0.25*(df["ApplicantIncome"]>4000).astype(int)*
    0.20*(df["LoanAmount"]<200).astype(int)+
    0.20*(df["Property_Area"]=="Urban").astype(int)
)
df["Loan_Status"]=np.where(np.random.rand(len(df))<approval_pb,"Y","N")
#Feature scaling
x=df.drop(["Loan_Status","Loan_ID"],axis=1)
y=df["Loan_Status"]
le=LabelEncoder()
y=le.fit_transform(y)
cat_cols=x.select_dtypes(include='object').columns.tolist()
num_cols=x.select_dtypes(include=["int64","float64"]).columns
num_cols=num_cols.tolist()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
#Making pipeline
preprocess=ColumnTransformer(
    transformers=[
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols),
        ("num",StandardScaler(),num_cols),
    ]
)
dt_clf=DecisionTreeClassifier(criterion="gini",
                              max_depth=6,
                              min_samples_leaf=10,
                              min_samples_split=20,
                              class_weight="balanced",
    random_state=42)
model=Pipeline(
    steps=[("preprocess",preprocess),
           ("model",dt_clf)]
)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy score is",accuracy_score(y_test,y_pred))
print("F1 score is",f1_score(y_test,y_pred))
print("Precision score is",precision_score(y_test,y_pred))
print("Recall score is",recall_score(y_test,y_pred))
print("Confusion Matrix is=",confusion_matrix(y_test,y_pred))
print("Train accuracy is",model.score(x_train,y_train))
print("Test accuracy is",model.score(x_test,y_test))
#Prediction
samples = [
    ["Male", "Yes", "0", "Graduate", "No", 5000, 0, 130, 360, 1, "Semiurban"],
    ["Female", "No", "2", "Not Graduate", "Yes", 2000, 0, 250, 180, 0, "Rural"],
    ["Male", "Yes", "1", "Graduate", "No", 7000, 2000, 180, 360, 1, "Urban"]
]

sample_df = pd.DataFrame(samples, columns=x_train.columns)

prei = model.predict(sample_df)
probs = model.predict_proba(sample_df)[:, 1]

for i in range(len(sample_df)):
    print(f"Sample {i+1}: Prediction: {prei[i]}, Probability: {probs[i]:.2f}")
