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
# 1. IMPORTATIONS 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

def entrainer_et_evaluer_rf(X_train, y_train, X_test, y_test):
    """
    Entraîne le modèle et affiche les résultats.
    """
    # Définition et initialisation
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Apprentissage
    rf_model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = rf_model.predict(X_test)
    
    # Affichage des scores
    print("Rapport de performance :")
    print(classification_report(y_test, y_pred))
    
    # Affichage graphique
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Blues', ax=ax)
    plt.title("Matrice de Confusion - Random Forest")
    plt.show()
    
    return rf_model

# 2. EXÉCUTION de ce ptn de code pour entraîner et évaluer le modèle
modele_rf = entrainer_et_evaluer_rf(X_train, y_train, X_test, y_test)
# %% [markdown]
# ## Interprétation de la Matrice de Confusion - Random Forest
# 
# La matrice de confusion permet de confronter les prédictions du modèle avec la réalité du terrain sur notre échantillon de test (998 individus).
# 
# ### 1. Les résultats chiffrés
# 
# * **Vrais Négatifs (487)** : Clients correctement identifiés comme fidèles.
# * **Vrais Positifs (479)** : Clients correctement identifiés comme étant sur le départ (Churn).
# * **Faux Positifs (8)** : Clients identifiés à tort comme partants (Erreurs de type I).
# * **Faux Négatifs (24)** : Clients partants que le modèle n'a pas réussi à détecter (Erreurs de type II).
# 
# ---
# 
# ### 2. Analyse des indicateurs de performance
# 
# Sur la base de ces résultats, nous calculons les métriques suivantes :
# 
# * **Précision globale (Accuracy)** : **96,8%**. Le modèle est globalement très fiable sur ce dataset de 3992 lignes.
# * **Précision du Churn** : **98,3%**. Lorsqu'il prédit un départ, le modèle a presque toujours raison. Cela évite d'envoyer des offres de rétention à des clients satisfaits.
# * **Rappel (Recall)** : **95,2%**. Le modèle parvient à détecter plus de 95% des clients qui vont réellement quitter la plateforme. C'est l'indicateur le plus critique pour Netflix.
# 
# ---
# 
# ### 3. Synthèse et impact métier
# 
# L'interprétation de ces résultats montre une excellente performance du Random Forest :
# 
# 1. **Optimisation des coûts** : Avec seulement 8 faux positifs, les campagnes de marketing ciblé ne gaspillent pas de budget.
# 2. **Réduction de l'attrition** : Le faible nombre de faux négatifs (24) signifie que la perte de revenus "surprise" est minimale.
# 
# Ce modèle servira de référence (baseline) pour comparer les performances de la prochaine étape : le réseau de neurones (Deep Learning).