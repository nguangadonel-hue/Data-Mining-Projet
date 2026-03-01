#%%
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

# %%
# %% [markdown]
# ## A quoi servent ces 3992 lignes ?
# Le chargement de notre fichier `netflix_churn_final.csv` nous donne un volume de **3992 individus**. 
#
# > **Note importante :** Ce chiffre est le socle de notre Phase 2. Il représente le dataset nettoyé et préparé pour l'entraînement de notre modèle de prédiction du churn. 
# > après avoir éliminé les erreurs de saisie (ex: temps de visionnage > 24h).
#
# ### Impact sur le modèle :
# 1. **Entraînement (80%)** : Environ 3193 lignes pour apprendre.
# 2. **Test (20%)** : Environ 799 lignes pour vérifier la précision.
# %%
# %% [markdown]
# ### Choix du modèle : Random Forest Classifier
# 
# Pour cette Phase 2, nous utilisons l'algorithme des Forêts Aléatoires (Random Forest). Ce choix se justifie par plusieurs points techniques :
# 1. **Robustesse** : Il réduit le risque de surapprentissage (overfitting) en combinant les prédictions de plusieurs arbres de décision.
# 2. **Importance des variables** : Il permet d'identifier quels facteurs (âge, abonnement, heures de visionnage) influencent le plus le départ des clients.
# 3. **Performance** : C'est l'un des modèles les plus efficaces pour les problèmes de classification binaire comme la prédiction du churn.
# 
# L'objectif est d'atteindre un équilibre entre la précision (ne pas se tromper de client) et le rappel (détecter un maximum de clients sur le départ).
# %%
