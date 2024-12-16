import pandas as pd

def load_data(file_path):
    """Charge les données et les nettoie en supprimant les colonnes vides"""
    data = pd.read_csv(file_path).dropna(axis=1)
    return data