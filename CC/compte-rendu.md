# JAMAL YASSINE

<img src="JAMAL YASSINE CAC2.jpg" style="height:464px;margin-right:432px"/>

# CAC2

# 22007655


# Compte rendu

## Analyse complète

# Détection de Fraude aux Cartes de Crédit

## Contexte
Dans un contexte de digitalisation accélérée des paiements, la fraude aux cartes de crédit représente un enjeu économique majeur pour les institutions financières, avec des pertes estimées à plusieurs milliards d'euros annuellement. Ce projet de Data Science, réalisé dans le cadre d'un module Machine Learning, vise à développer un système prédictif capable d'identifier les transactions frauduleuses en temps réel à partir du dataset Kaggle "Credit Card Fraud Detection" (284 807 transactions européennes sur 2 jours, 0,172% de fraudes).

## Problématique
Classification binaire supervisée sur données hautement déséquilibrées : prédire la variable cible 'Class' (0=légitime, 1=fraude) en exploitant 28 features anonymisées (PCA), 'Time' et 'Amount'. L'objectif est de minimiser les faux négatifs tout en optimisant Precision/Recall via SMOTE, cross-validation et hyperparamétrage. Thématique : Économie/Finance. 

## Méthodologie
- **Preprocessing** : Nettoyage, feature engineering (Amount_log), RobustScaler
- **EDA** : Visualisations distributions/corrélations, interprétations
- **Modélisation** : 3 algorithmes (LogisticRegression, RandomForest, XGBoost) + GridSearchCV
- **Évaluation** : ROC-AUC, F1-Score, matrice de confusion

- # 📄 Compte rendu – Détection de fraude sur transactions bancaires par apprentissage supervisé

## 1. 📁 À propos du jeu de données  
Le travail repose sur le dataset **Credit Card Fraud Detection** provenant de Kaggle, composé de **284 807 transactions** décrites par **31 variables** :  
- **28 variables anonymisées** (`V1`–`V28`), issues d’une transformation PCA ;  
- **2 variables originales** (`Time` et `Amount`) ;  
- **1 variable cible (`Class`)** indiquant :  
  - `0` = transaction légitime,  
  - `1` = transaction frauduleuse.  

Le dataset est **extrêmement déséquilibré**, ne contenant que **492 fraudes** pour **284 315 transactions normales**, soit un taux de fraude d’environ **0,17 %**.  
Ce déséquilibre impose l’utilisation de techniques adaptées pour l’apprentissage supervisé.

---

## 2. 🎯 Introduction et contexte  
La détection de fraude bancaire constitue un enjeu crucial pour les institutions financières, qui doivent identifier rapidement les transactions suspectes tout en limitant les fausses alertes.  

L’objectif de ce projet est de construire un modèle d’apprentissage supervisé capable de :  
- détecter efficacement les transactions frauduleuses,  
- réduire les pertes financières associées aux fraudes non détectées,  
- maintenir un niveau faible de faux positifs pour préserver l’expérience client.

Dans un contexte de classes très déséquilibrées, les métriques traditionnelles comme l’accuracy sont **insuffisantes**.  
Les indicateurs prioritaires sont :  
- **Recall**, pour éviter les faux négatifs,  
- **Précision**,  
- **F1-score**,  
- **ROC-AUC**, adapté aux déséquilibres extrêmes.

---

## 3. 📊 Analyse exploratoire (EDA)  
L’analyse exploratoire réalisée confirme les éléments clés suivants :

### ✔ Déséquilibre massif  
La classe `1` représente moins de 1 transaction sur 500.

### ✔ Variables PCA  
Les composantes `V1` à `V28` sont déjà centrées-réduites.  
Certaines variables (ex. `V14`, `V17`) montrent des distributions distinctes entre fraudes et non-fraudes, suggérant une bonne séparabilité.

### ✔ Montant des transactions  
`Amount` présente une distribution très asymétrique.  
Une transformation logarithmique est pertinente pour réduire cette asymétrie.

### ✔ Corrélations  
La matrice de corrélation montre très peu de relations linéaires fortes en raison de la PCA, mais certaines variables se démarquent dans les cas de fraude.

---

## 4. 🔧 Préparation et ingénierie des données

### ✔ Suppression des doublons  
Les doublons détectés ont été supprimés pour éviter un biais dans l'apprentissage.

### ✔ Création de nouvelles variables  
À partir de `Amount`, deux nouvelles caractéristiques utiles ont été ajoutées :  
- `Amount_Scaled` (scalée via `RobustScaler`),  
- `Log_Amount` (transformation logarithmique).

### ✔ Standardisation  
Les colonnes `Time` et `Amount` brutes ont été retirées, car leur version transformée est plus pertinente pour la modélisation.

### ✔ Découpage du dataset  
Un split **80 % / 20 %** a été réalisé avec **stratification sur `Class`** afin de conserver la proportion de fraudes dans chaque sous-échantillon.

---

## 5. 🤖 Méthodologie de modélisation

Trois algorithmes supervisés ont été étudiés :  
- **Régression Logistique**,  
- **Random Forest**,  
- **XGBoost**.

### ✔ Gestion du déséquilibre  
La technique **SMOTE** est utilisée dans un pipeline pour sur-échantillonner la classe minoritaire **uniquement sur les données d’entraînement**, évitant toute fuite d’information.  

### ✔ Validation croisée et optimisation  
Chaque modèle est intégré dans un pipeline comprenant :  
- standardisation des données,  
- oversampling (SMOTE),  
- classification.

La recherche d’hyperparamètres est réalisée via **GridSearchCV**, avec comme scoring principal :  
➡️ `roc_auc`, adapté au déséquilibre extrême.

Cette configuration permet une évaluation robuste et cohérente de chaque modèle.

---

## 6. 📈 Résultats, limites et recommandations

### ✔ Résultats observés  
Les premiers tests montrent que :  
- l’accuracy n’est pas pertinente (trop influencée par la classe majoritaire),  
- le **ROC-AUC** est nettement plus représentatif des performances,  
- les métriques clés pour la classe de fraude sont le **Recall**, la **Précision** et le **F1-Score**.

Les modèles avancés comme **Random Forest** et **XGBoost** montrent un fort potentiel pour améliorer la détection des fraudes.

### ✔ Limites rencontrées  
- Configuration initiale du scoring dans GridSearch nécessitant une correction (`scoring="roc_auc"`).  
- Faible nombre relatif de fraudes entraînant une variabilité élevée sur les mesures de performances.  
- Les variables PCA ne permettent pas une interprétation métier directe.

### ✔ Recommandations  
- Finaliser l’optimisation des hyperparamètres.  
- Explorer l’utilisation d’algorithmes supplémentaires (Isolation Forest, modèles neuronaux).  
- Générer un tableau comparatif complet des résultats (AUC, Recall, F1…).  
- Mettre en place un modèle en production avec seuil de décision ajustable.

---

## 7. 🏁 Conclusion  
Ce projet illustre les défis de la détection de fraude sur des données massives et fortement déséquilibrées.  
La chaîne d’analyse mise en place — nettoyage, ingénierie de variables, rééquilibrage, validation croisée — établit une base solide pour sélectionner le modèle le plus performant.
 
- finaliser le tuning des modèles,  
- comparer leurs performances avec des métriques robustes,  
- choisir la solution offrant le meilleur compromis entre détection des fraudes et réduction des faux positifs.

Ce travail constitue une avancée significative vers la création d’un système fiable de détection de fraude bancaire.


