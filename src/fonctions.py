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

def final_model_training(X, y):
    """Entraîne les modèles sur toutes les données disponibles"""
    final_svm_model = SVC()
    final_nb_model = GaussianNB()
    final_rf_model = RandomForestClassifier(random_state=18)
    
    final_svm_model.fit(X, y)
    final_nb_model.fit(X, y)
    final_rf_model.fit(X, y)
    return final_svm_model, final_nb_model, final_rf_model

def evaluate_test_data(test_data_path, final_svm_model, final_nb_model, final_rf_model, encoding):
    """Évalue les modèles sur un jeu de test externe"""
    test_data = pd.read_csv(test_data_path).dropna(axis=1)
    test_X = test_data.iloc[:, :-1]
    test_Y = encoding.transform(test_data.iloc[:, -1])
    
    svm_preds = final_svm_model.predict(test_X)
    nb_preds = final_nb_model.predict(test_X)
    rf_preds = final_rf_model.predict(test_X)

    final_preds = [stats.mode([i, j, k])[0] for i, j, k in zip(svm_preds, nb_preds, rf_preds)]

    print(f"Accuracy sur l'ensemble des données de test: {accuracy_score(test_Y, final_preds)*100}")

    cf_matrix = confusion_matrix(test_Y, final_preds)
    plt.figure(figsize=(12,8))
    sns.heatmap(cf_matrix, annot=True)
    plt.title("Matrice de confusion de l'ensemble des données test")
    plt.show()

def create_symptom_dict(X):
    """Crée un dictionnaire des symptômes avec leur index"""
    symptoms = X.columns.values
    symptom_index = { " ".join([i.capitalize() for i in value.split("_")]): index for index, value in enumerate(symptoms) }
    return symptom_index