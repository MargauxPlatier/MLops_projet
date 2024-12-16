import pandas as pd

def load_data_train(file_path):
    """Charge les données et les nettoie en supprimant les colonnes vides"""
    data_train = pd.read_csv(file_path).dropna(axis=1)
    return data_train

def load_data_test(file_path):
    """Charge les données et les nettoie en supprimant les colonnes vides"""
    data_test = pd.read_csv(file_path).dropna(axis=1)
    return data_test