import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. On définit la fonction avec un nom de variable générique 'file_path'
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    X = df.drop('churned', axis=1)
    y = df['churned']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# 2. C'est ici, tout en bas, qu'on appelle la fonction avec le VRAI nom du fichier
if __name__ == "__main__":
    # Puisque le fichier est dans le même dossier, le nom suffit
    NOM_FICHIER = 'netflix_churn_final.csv' 
    X_train, X_test, y_train, y_test = load_and_preprocess(NOM_FICHIER)
    print(f"Données prêtes : {X_train.shape[0]} lignes chargées.")
