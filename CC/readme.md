# JAMAL YASSINE

<img src="JAMAL YASSINE CAC2.jpg" style="height:464px;margin-right:432px"/>

# CAC2

# 22007655


# Compte rendu

## Analyse complète

# Détection de Fraude aux Cartes de Crédit

## Contexte
Dans un contexte de digitalisation accélérée des paiements, la fraude aux cartes de crédit représente un enjeu économique majeur pour les institutions financières, avec des pertes estimées à plusieurs milliards d'euros annuellement. Ce projet de Data Science, réalisé dans le cadre d'un module Machine Learning, vise à développer un système prédictif capable d'identifier les transactions frauduleuses en temps réel à partir du dataset Kaggle "Credit Card Fraud Detection" (284 807 transactions européennes sur 2 jours, 0,172% de fraudes). [web:16][file:1]

## Problématique
Classification binaire supervisée sur données hautement déséquilibrées : prédire la variable cible 'Class' (0=légitime, 1=fraude) en exploitant 28 features anonymisées (PCA), 'Time' et 'Amount'. L'objectif est de minimiser les faux négatifs tout en optimisant Precision/Recall via SMOTE, cross-validation et hyperparamétrage. Thématique : Économie/Finance. [file:1]

## Méthodologie
- **Preprocessing** : Nettoyage, feature engineering (Amount_log), RobustScaler
- **EDA** : Visualisations distributions/corrélations, interprétations
- **Modélisation** : 3 algorithmes (LogisticRegression, RandomForest, XGBoost) + GridSearchCV
- **Évaluation** : ROC-AUC, F1-Score, matrice de confusion [file:1]
