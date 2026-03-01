# %% [Markodwn]
# Projet de Prediction du Churn Netflix 

Ce projet identifie les facteurs de résiliation (churn) des clients Netflix et compare l'efficacité de deux approches de Machine Learning : la Forêt Aléatoire (Random Forest) et le Deep Learning.

---

## 1. Architecture du Code (Modularite)

Le script `predict_churn_netflix.py` est structuré en fonctions indépendantes pour assurer la clarté du pipeline et répondre aux critères de modularité du projet :

* **load_and_preprocess** : Chargement du dataset (3992 lignes), nettoyage des données, encodage des variables et normalisation via StandardScaler.
* **entrainer_et_evaluer_rf** : Entraînement du modèle Random Forest et génération de la matrice de confusion pour l'évaluation.
* **analyser_importance_variables** : Extraction des caractéristiques clés pour identifier les leviers réels de désabonnement.
* **entrainer_deep_learning** : Implémentation d'un réseau de neurones (Perceptron Multicouche) avec régularisation par Dropout.

---

## 2. Installation et Execution

### Installation des dependances
```bash
pip install pandas scikit-learn matplotlib seaborn tensorflow joblib

## 3. Resultats et Performances

L'evaluation a ete realisee sur un echantillon de test de 998 individus (25% du dataset total).

| Metrique | Random Forest | Deep Learning |
| :--- | :--- | :--- |
| **Precision (Accuracy)** | **96,8%** | 90,2% |
| **Rappel (Recall)** | **95,2%** | 88,4% |
| **F1-Score** | **96,0%** | 89,3% |



---

## 4. Analyse Metier et Recommandations

L'analyse de l'importance des variables confirme que le churn est pilote par des facteurs comportementaux plutot que demographiques :

* **last_login_days** : Le delai depuis la derniere connexion est l'indicateur le plus critique. Un allongement de ce delai precede systematiquement la resiliation.
* **watch_hours** : Une baisse significative du temps de visionnage hebdomadaire constitue un signal precurseur de desengagement.

**Conclusion** : Nous preconisons l'utilisation de la **Random Forest**. Ce modele offre une precision superieure sur ce volume de donnees et permet une interpretabilite directe des leviers d'action, contrairement au reseau de neurones qui agit ici comme une boite noire.

