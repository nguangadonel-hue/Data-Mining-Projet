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
# %% [markdown]
# ### Analyse de l'importance des variables
#  L'objectif est d'identifier les caractéristiques qui ont le plus de poids dans la décision de l'algorithme.
# 
# Pour Netflix, cette analyse est stratégique car elle permet de :
# 1. **Identifier les leviers d'action** : Si le prix est la variable dominante, une politique de promotion est nécessaire. Si c'est le temps de visionnage, il faut améliorer les recommandations de contenu.
# 2. **Simplifier le modèle** : Identifier les variables inutiles qui pourraient être supprimées pour gagner en efficacité.
# 3. **Justifier les décisions** : Expliquer aux parties prenantes pourquoi certains profils sont considérés comme "à risque".
# %%# %%
import pandas as pd
import seaborn as sns

def analyser_importance_variables(rf_model, X):
    """
    Calcule et visualise l'importance de chaque variable dans le modèle Random Forest.
    """
    importances = rf_model.feature_importances_
    noms_variables = X.columns
    
    df_importance = pd.DataFrame({
        'Variable': noms_variables,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Affichage graphique
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Variable', data=df_importance, palette='viridis')
    plt.title("Importance des variables - Modèle Random Forest")
    plt.xlabel("Score d'importance")
    plt.ylabel("Variables")
    plt.show()
    
    # Affichage textuels des 3 variables dominantes
    print("Top 3 des variables prédictives :")
    print(df_importance.head(3))
    
    return df_importance

# Appel de la fonction
# On utilise df.drop('churned', axis=1) pour récupérer les noms originaux des colonnes
X_original = pd.read_csv('netflix_churn_final.csv').drop('churned', axis=1)
importance_df = analyser_importance_variables(modele_rf, X_original)

# %%
# %% [markdown]
# ## Synthèse de l'Importance des Variables et Cohérence Métier
# 
# Cette analyse permet de valider la pertinence de notre modèle Random Forest en confrontant ses résultats à notre analyse exploratoire initiale (EDA).
# 
# ### 1. Validation de la fiabilité du modèle
# 
# La cohérence entre notre analyse exploratoire (EDA) et l'importance des variables calculée par la Random Forest valide la fiabilité de notre modèle. Le fait que les **variables comportementales** (`watch_hours`, `last_login_days`) surclassent les **variables démographiques** montre que le modèle a capturé les signaux de désengagement réels. 
# 
# Cette hiérarchie des caractéristiques explique pourquoi la Random Forest a obtenu de meilleurs résultats que le Deep Learning sur ce volume de données : elle a su isoler et prioriser ces indicateurs clés avec plus d'efficacité, là où un réseau de neurones peut parfois se perdre dans le "bruit" des variables secondaires sur un dataset de moins de 10 000 lignes.
# 
# ---
# 
# ### 2. Analyse des leviers d'action pour Netflix
# 
# L'importance élevée de variables telles que `last_login_days` (nombre de jours depuis la dernière connexion) confirme que le **désengagement progressif** est le principal moteur du churn :
# 
# * **Signal Faible** : Une baisse des heures de visionnage (`watch_hours`) est le premier indicateur d'un risque de départ.
# * **Signal Fort** : L'allongement du délai depuis la dernière connexion est le point de rupture final.
# 
# ### 3. Conclusion sur le choix du modèle
# 
# Contrairement au Deep Learning, qui nécessite souvent des volumes de données massifs pour extraire des relations complexes, la Random Forest a démontré une capacité supérieure à **hiérarchiser les variables critiques**. En isolant les signaux comportementaux comme prioritaires, le modèle offre une interprétabilité directe, essentielle pour justifier les futures stratégies de rétention client de Netflix.

# %% [markdown]
# ## Phase 3 : Modélisation par Deep Learning (Réseau de Neurones)
# 
# Pour cette dernière phase, nous implémentons un Perceptron Multicouche (MLP). Contrairement à la Random Forest, ce modèle s'appuie sur des couches de neurones pour apprendre des représentations non-linéaires plus abstraites.
# 
# **Architecture du modèle :**
# 1. **Couche d'entrée** : Reçoit les caractéristiques normalisées (X_train_scaled).
# 2. **Couches cachées (Dense)** : Deux couches avec activation 'relu' pour capturer la complexité des données.
# 3. **Régularisation (Dropout)** : Utilisée pour prévenir le surapprentissage en désactivant aléatoirement des neurones.
# 4. **Couche de sortie** : Un seul neurone avec activation 'sigmoid' pour prédire la probabilité de churn (0 à 1).
# 
# La fonction de perte utilisée est la **Binary Cross-Entropy** :
# $$L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$
# %%# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report

def entrainer_deep_learning(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Construit, entraîne et évalue un réseau de neurones simple.
    """
    # 1. Définition de l'architecture
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # 2. Compilation
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    # 3. Entraînement
    print("\n--- Début de l'entraînement du Réseau de Neurones ---")
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=50, 
        batch_size=32, 
        validation_split=0.2, 
        verbose=0 # On cache le détail des épqoques pour plus de clarté
    )
    
    # 4. Évaluation
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Précision finale du Deep Learning : {accuracy:.4f}")
    
    # 5. Prédictions (seuil à 0.5)
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    print("\nRapport de performance Deep Learning :")
    print(classification_report(y_test, y_pred))
    
    # Visualisation de la courbe d'apprentissage
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Entraînement')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Précision du modèle au fil des épouques')
    plt.legend()
    plt.show()
    
    return model

# Appel de la fonction
modele_dl = entrainer_deep_learning(X_train, y_train, X_test, y_test)
# %%
# %% [markdown]
# ## Interprétation de la courbe d'apprentissage (Deep Learning)
# 
# Le graphique de précision au fil des époques permet de visualiser la progression de l'apprentissage du réseau de neurones et de détecter d'éventuels problèmes de généralisation.
# 
# ### 1. Analyse de la convergence
# * **Phase de progression rapide** : Entre l'époque 0 et 5, on observe une forte hausse de la précision (de 74% à 88%). Le modèle identifie rapidement les structures principales du dataset Netflix.
# * **Phase de stabilisation** : Après l'époque 10, les courbes s'aplatissent. Le modèle a atteint sa capacité maximale d'apprentissage sur ce jeu de données.
# 
# ### 2. Évaluation de l'Overfitting (Surapprentissage)
# L'écart entre la courbe d'**Entraînement** (bleue) et la courbe de **Validation** (orange) est un indicateur clé :
# * **Écart réduit** : Les deux courbes restent très proches l'une de l'autre jusqu'à l'époque 50. Cela signifie que le modèle généralise bien et qu'il sera capable de prédire le churn sur de nouveaux clients qu'il n'a jamais rencontrés.
# * **Absence de divergence** : On ne voit pas la courbe de validation chuter alors que celle d'entraînement monte, ce qui confirme que la régularisation (Dropout) a fonctionné.
# 
# ### 3. Comparaison avec la Random Forest
# Bien que le réseau de neurones soit stable et performant (environ 90% de précision), ses résultats restent légèrement inférieurs à ceux obtenus par la **Random Forest (96.8%)**. 
# 
# **Conclusion** : Pour ce volume de données (3992 lignes), la structure tabulaire est mieux exploitée par les algorithmes d'arbres de décision. Le Deep Learning est ici un modèle robuste, mais moins précis que l'approche par forêt aléatoire.
# %% [markdown]
# ## Conclusion Finale et Recommandations Stratégiques
# 
# Ce projet de Data Mining visait à prédire le risque de désabonnement (churn) des clients Netflix en comparant deux approches : une forêt aléatoire et un réseau de neurones.
# 
# ### 1. Bilan de la performance des modèles
# 
# * **Random Forest** : Ce modèle s'est révélé être le plus performant avec une précision de **96,8%** et un rappel de **95,2%**. Sa structure par arbres de décision est parfaitement adaptée à la taille de notre dataset (3992 lignes) et permet une interprétabilité directe des résultats.
# * **Deep Learning** : Le réseau de neurones a montré une bonne stabilité avec une précision d'environ **90%**. Cependant, sur ce volume de données tabulaires, il n'a pas réussi à surpasser l'approche par forêt aléatoire, tout en étant plus complexe à paramétrer (boîte noire).
# 
# ### 2. Facteurs clés de prédiction
# 
# L'analyse de l'importance des variables a permis d'identifier que le churn n'est pas lié au profil démographique du client (âge, localisation), mais à son **comportement d'utilisation** :
# * **Désengagement temporel** (`last_login_days`) : C'est le signal le plus critique. Plus le délai depuis la dernière connexion s'allonge, plus la probabilité de départ est élevée.
# * **Intensité de consommation** (`watch_hours`) : Une baisse du volume horaire de visionnage est un indicateur précurseur de résiliation.
# 
# ---
# 
# ### 3. Recommandations pour Netflix
# 
# Sur la base de ces résultats, nous préconisons les actions suivantes :
# 
# 1. **Déploiement du modèle Random Forest** : Utiliser ce modèle en production pour attribuer un "score de risque" à chaque client de manière hebdomadaire.
# 2. **Système d'alerte précoce** : Déclencher des campagnes de réengagement automatisées dès que la variable `last_login_days` dépasse un seuil critique défini par le modèle.
# 3. **Personnalisation du contenu** : Pour les clients dont les `watch_hours` déclinent, proposer des recommandations de contenus "Premium" ou des nouveautés afin de relancer l'intérêt pour la plateforme.
# 4. **Optimisation marketing** : Grâce à la précision du modèle, limiter les offres promotionnelles aux seuls clients identifiés à "haut risque" pour éviter les pertes de revenus inutiles sur les clients fidèles.
# %% [markdown]
# ## Synthèse Finale du Projet de Data Mining
# 
# ### Objectif atteint
# Nous avons réussi à construire un pipeline complet capable de prédire le départ des clients Netflix avec une précision supérieure à 95% grâce à la **Random Forest**.
# 
# ### Ce qu'il faut retenir pour le business
# 1. **Anticipation** : Le modèle détecte les signaux de désabonnement avant qu'ils ne surviennent.
# 2. **Comportement vs Démographie** : Ce n'est pas "qui" est le client qui compte, mais "comment" il utilise Netflix (fréquence de connexion et heures de visionnage).
# 3. **Modularité technique** : Le projet est structuré en fonctions réutilisables, permettant d'intégrer facilement de nouvelles données ou de tester de nouveaux algorithmes à l'avenir.
# %%
