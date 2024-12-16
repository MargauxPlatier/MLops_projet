import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import statistics


def split_data(data):
    """Divise les données en ensemble d'entraînement et de test"""
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    return X_train, X_test, y_train, y_test

def cv_scoring(estimator, X, y):
    """Fonction de scoring pour la validation croisée"""
    return accuracy_score(y, estimator.predict(X))

def cross_validate_models(models, X, y):
    """Évalue les modèles avec une validation croisée k-fold"""
    for model_name, model in models.items():
        scores = cross_val_score(model, X, y, cv=10, n_jobs=-1, scoring=cv_scoring)
        print("="*60)
        print(f"{model_name}")
        print(f"Scores: {scores}")
        print(f"Mean Score: {np.mean(scores)}")

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """Entraîne un modèle et évalue sa performance"""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy_train = accuracy_score(y_train, model.predict(X_train)) * 100
    accuracy_test = accuracy_score(y_test, preds) * 100
    print(f"Accuracy sur les données d'entrainement : {accuracy_train}")
    print(f"Accuracy sur les données de test : {accuracy_test}")
    
    cf_matrix = confusion_matrix(y_test, preds)
    plt.figure(figsize=(12,8))
    sns.heatmap(cf_matrix, annot=True)
    plt.title(f"Matrice de confusion pour {model.__class__.__name__}")
    plt.show()
