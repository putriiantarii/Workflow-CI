# -*- coding: utf-8 -*-
"""
Modelling Tuning (Skilled/Advanced)

Pada tahap ini, saya membangun model machine learning menggunakan dataset
yang telah melalui preprocessing (heart_preprocessing.csv).

Tujuan dari proses ini adalah melakukan hyperparameter tuning pada model
Random Forest untuk mendapatkan model terbaik, serta melakukan manual
logging ke MLflow Tracking untuk mencatat metrics dan artefak.
"""

#1. Import Library
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#2. Load Dataset
# Pastikan file heart_preprocessing.csv sudah ada di folder MLProject
df = pd.read_csv("heart_preprocessing.csv")

#3. Split Data
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#4. Hyperparameter Tuning

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, None],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy"
)


# 5. Manual MLflow Logging
with mlflow.start_run():

    # Train model
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Log hyperparameter
    mlflow.log_params(grid.best_params_)

    # Evaluation
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    #Artifak 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    #Artifak 2: ROC Curve
    y_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_artifact("roc_curve.png")

    #Artifak 3: Feature Importance
    result = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=10, random_state=42
    )

    importances = result.importances_mean
    np.savetxt("feature_importance.txt", importances)
    mlflow.log_artifact("feature_importance.txt")

    # Log model
    mlflow.sklearn.log_model(best_model, "model")


print("Training dan MLflow logging selesai.")
