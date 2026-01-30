#Importing the modules
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,recall_score
#Reading and printing the dataset
spam_messages = [
    "Congratulations! You have won a free lottery ticket",
    "Urgent! Your account has been compromised",
    "Win cash prizes now, click here",
    "Limited offer! Buy now and save big",
    "You are selected for a free gift card",
    "Earn money from home easily",
    "Exclusive deal just for you",
    "Claim your reward immediately",
    "Act now! Offer expires today",
    "You have won a cash reward"
]

ham_messages = [
    "Hi, are we meeting tomorrow?",
    "Please find the attached report",
    "Let's catch up over coffee",
    "Can you review my code?",
    "Meeting scheduled at 10 AM",
    "I will call you later",
    "Project deadline is next week",
    "Thanks for your help",
    "Please update me on the status",
    "Let us know your availability"
]

data = []

for _ in range(1000):
    if random.random() > 0.5:
        data.append([random.choice(spam_messages), "spam"])
    else:
        data.append([random.choice(ham_messages), "ham"])

df = pd.DataFrame(data, columns=["email_text", "label"])
df.to_csv("email_spam_large_dataset.csv", index=False)
print("Dataset created successfully!")
print(df.head())
print(df.shape)
print(df.describe())
print(df.isnull().sum())
print(df.info())
#Feature scaling
x=df["email_text"]
y=df["label"]
le=LabelEncoder()
y=le.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
#Making pipeline
model=Pipeline([
    ("tfidf",TfidfVectorizer(lowercase=True,stop_words='english')),
     ("nb",MultinomialNB())
])
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#Checking the accuracy,f1,recall and precision
accuracy=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred,average="binary")
print("Accuracy:",accuracy)
print("F1:",f1)
print("Recall:",recall_score(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))