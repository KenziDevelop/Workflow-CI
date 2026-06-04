# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri('http://localhost:5000')

mlflow.autolog()

if os.path.exists("saved_model"):
    shutil.rmtree("saved_model")

df = pd.read_csv('heart_preprocessing.csv')

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy_score", acc)
    mlflow.log_param("n_estimators_manual", 50)

    plt.figure(figsize=(4,3))
    plt.text(0.5, 0.5, f'Akurasi Model: {acc:.2f}', ha='center', va='center')
    plt.axis('off')
    plt.savefig("akurasi.png")
    mlflow.log_artifact("akurasi.png")

    mlflow.sklearn.log_model(rf, "model")
    mlflow.sklearn.save_model(rf, "saved_model")

print("Training selesai dan tercatat di MLflow!")
