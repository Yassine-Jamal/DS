# JAMAL YASSINE

<img src="JAMAL YASSINE CAC2.jpg" style="height:464px;margin-right:432px"/>

# CAC2

# 22007655


# Compte rendu

## Analyse complète de la base de données "Instagram Analytics Dataset"

# À propos de ce jeu de donnée :
Ce fichier contient 30 000 publications Instagram avec des analyses détaillées collectées au cours des 12 derniers mois. Chaque ligne représente une publication Instagram et inclut des informations sur le type de média, les indicateurs d'engagement, la portée, les impressions, les enregistrements, les partages, les sources de trafic et le taux d'engagement estimé.

Ce jeu de données est conçu pour simuler des données Instagram Insights réalistes et reproduit le comportement naturel de l'algorithme d'Instagram. Des indicateurs tels que les mentions « J'aime », la portée, les impressions, les enregistrements et le nombre d'abonnés gagnés ont été générés à l'aide de distributions statistiques réalistes afin de refléter les performances typiques des publications Photos, Vidéos, Reels et Carrousel.

Ce fichier est idéal pour explorer :
quel type de contenu est le plus performant ;
le lien entre la portée, les impressions et l’engagement ;
les sources de trafic (Explorer, Flux Reels, Hashtags, etc.) qui génèrent de la visibilité ; l’influence
de la longueur des légendes et des hashtags sur la visibilité
; la prédiction du taux d’engagement et des tendances de croissance ;
la modélisation des facteurs de succès sur Instagram grâce à l’apprentissage automatique.

Ce fichier ne contient aucune donnée réelle d'utilisateurs Instagram. Il est entièrement synthétique et peut être utilisé sans danger pour la recherche publique, les études universitaires, les compétitions Kaggle et les projets d'analyse des réseaux sociaux.


# Prédiction du Taux d’Engagement Instagram

Analyse de données Instagram et comparaison de modèles de régression pour prédire le taux d’engagement des publications.

---

## 📑 Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)  
2. [Analyse Exploratoire des Données](#2-analyse-exploratoire-des-données)  
   2.1 [Chargement et Structure du Dataset](#21-chargement-et-structure-du-dataset)  
   2.2 [Prétraitement et Ingénierie de Caractéristiques](#22-prétraitement-et-ingénierie-de-caractéristiques)  
   2.3 [Gestion des Valeurs Manquantes](#23-gestion-des-valeurs-manquantes)  
   2.4 [Analyse Statistique et Visuelle](#24-analyse-statistique-et-visuelle)  
3. [Méthodologie de Modélisation](#3-méthodologie-de-modélisation)  
   3.1 [Séparation des Données (Train/Test)](#31-séparation-des-données-traintest)  
   3.2 [Modèles de Régression Testés](#32-modèles-de-régression-testés)  
4. [Résultats et Comparaison des Modèles](#4-résultats-et-comparaison-des-modèles)  
   4.1 [Régression Linéaire](#41-régression-linéaire)  
   4.2 [Régression Polynomiale](#42-régression-polynomiale)  
   4.3 [Régression par Arbre de Décision](#43-régression-par-arbre-de-décision)  
   4.4 [Régression par Forêt Aléatoire](#44-régression-par-forêt-aléatoire)  
   4.5 [Régression SVR](#45-régression-svr)  
   4.6 [Tableau Comparatif des Performances](#46-tableau-comparatif-des-performances)  
5. [Analyse des Résultats et Recommandations](#5-analyse-des-résultats-et-recommandations)  
6. [Conclusion](#6-conclusion)  


---

## 1. Introduction et Contexte

Ce projet présente une analyse détaillée d’un jeu de données réel concernant les statistiques d’engagement de publications Instagram.  
L’objectif est de construire et comparer plusieurs **modèles de régression** afin de prédire le **taux d’engagement** (`engagement_rate`) à partir de différentes caractéristiques liées :

- au contenu (type, catégorie, texte),
- à l’audience (reach, impressions, followers gagnés),
- à l’interaction (likes, commentaires, partages, enregistrements),
- à la temporalité (date/heure de publication).

Le pipeline suit les étapes classiques d’un projet de Data Science :

- Analyse exploratoire (EDA),
- Prétraitement et ingénierie de caractéristiques,
- Modélisation et évaluation,
- Comparaison et sélection du meilleur modèle.

---

## 2. Analyse Exploratoire des Données

### 2.1 Chargement et Structure du Dataset

Le fichier principal est `Instagram_Analytics.csv`.

- Nombre d’observations : **29 999** publications  
- Nombre de variables : **15** colonnes (14 features + 1 cible)

**Variable cible (Y)**  
- `engagement_rate` : taux d’engagement (en pourcentage)

**Variables d’entrée (X)** (exemples) :

- Engagement direct : `likes`, `comments`, `shares`, `saves`
- Portée / audience : `reach`, `impressions`, `followers_gained`
- Métadonnées texte : `caption_length`, `hashtags_count`
- Temporelle : `upload_date`
- Catégorielles :
  - `media_type` (Reel, Photo, Video, Carousel)
  - `traffic_source` (Home Feed, Hashtags, Reels Feed, External, Profile, Explore)
  - `content_category` (Technology, Fitness, Beauty, Music, Travel, Photography, etc.)

Un premier aperçu (`df.shape`, `df.info()`, `df.head()`) permet de confirmer la cohérence du fichier et l’absence de types anormaux.

### 2.2 Prétraitement et Ingénierie de Caractéristiques

1) Conversion de la date en **datetime** et extraction de caractéristiques temporelles :

- `upload_year`
- `upload_month`
- `upload_day_of_week` (0 = Lundi, ..., 6 = Dimanche)
- `upload_hour`

Ces nouvelles features capturent l’impact potentiel du moment de publication sur l’engagement.

2) Encodage des variables catégorielles via **One-Hot Encoding** pour :

- `media_type`
- `traffic_source`
- `content_category`

L’option `drop_first=True` est utilisée pour réduire la multicolinéarité (suppression d’une catégorie de référence par variable).

3) Création d’un DataFrame nettoyé :

- Suppression de `post_id` (identifiant sans valeur prédictive),
- Suppression de `upload_date` brute (remplacée par les features temporelles),
- Conservation uniquement de features **numériques** dans `df_processed`.

### 2.3 Gestion des Valeurs Manquantes

Une vérification systématique est réalisée :

- `df_processed.isnull().sum()`

Résultat : aucune valeur manquante détectée → **aucune imputation nécessaire**, ce qui simplifie la suite de la modélisation.

### 2.4 Analyse Statistique et Visuelle

Quelques points clefs :

- La distribution de `engagement_rate` est légèrement asymétrique, avec la majorité des valeurs dans une plage “modérée” et quelques valeurs extrêmes.
- Des visualisations (histogrammes, boxplots, pairplots, heatmap de corrélation) sont utilisées pour :
  - explorer les relations entre `engagement_rate` et les variables d’engagement (likes, comments, shares, reach, etc.),
  - identifier des corrélations et des patterns intéressants.
- Les échelles des variables (`likes`, `reach`, `impressions`, etc.) sont très différentes → cela motive l’usage d’une **normalisation / standardisation** pour les modèles sensibles à l’échelle (comme SVR).

---

## 3. Méthodologie de Modélisation

### 3.1 Séparation des Données (Train/Test)

Séparation standard en jeu d’entraînement et jeu de test :

- Cible :  
  `y = df_processed['engagement_rate']`
- Features :  
  `X = df_processed.drop(columns=['engagement_rate'])`

Split :

- 80 % pour l’**entraînement** (`X_train`, `y_train`)
- 20 % pour le **test** (`X_test`, `y_test`)
- `random_state=42` pour assurer la reproductibilité

### 3.2 Modèles de Régression Testés

Cinq modèles de régression ont été entraînés et évalués :

1. Régression Linéaire
2. Régression Polynomiale (degré 2)
3. Régression par Arbre de Décision
4. Régression par Forêt Aléatoire (Random Forest)
5. Régression SVR (Support Vector Regression, noyau RBF, avec normalisation préalable)

Les performances sont évaluées selon trois métriques :

- **R²** (coefficient de détermination)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)

---

## 4. Résultats et Comparaison des Modèles

### 4.1 Régression Linéaire

Modèle de base supposant une relation linéaire entre les features et la cible.

- R² ≈ 0.0899 (≈ 9 % de variance expliquée)
- MSE ≈ 2238.45
- RMSE ≈ 47.31

Conclusion : le modèle explique très peu la variance, ce qui suggère que la relation entre les variables d’entrée et le taux d’engagement est fortement **non linéaire**.

### 4.2 Régression Polynomiale

Régression polynomiale de **degré 2** (ajout de termes quadratiques et d’interactions).

- R² ≈ 0.1706 (≈ 17 %)
- MSE ≈ 2062.18
- RMSE ≈ 45.41

La performance est légèrement meilleure que la régression linéaire, mais reste insuffisante au regard de la complexité du problème.

### 4.3 Régression par Arbre de Décision

Modèle non paramétrique, basé sur des partitions récursives de l’espace des features.

- R² ≈ 0.7126 (≈ 71 %)
- MSE ≈ 707.89
- RMSE ≈ 26.61

C’est le **meilleur modèle** parmi ceux testés :

- Forte hausse de R² par rapport aux modèles linéaires,
- Réduction importante de l’erreur (RMSE).

Cela montre que les arbres capturent très bien la nature non linéaire et les interactions complexes entre les variables.

### 4.4 Régression par Forêt Aléatoire

Ensemble de nombreux arbres de décision, entraînés sur des sous-échantillons et sous-ensembles de variables.

- R² ≈ 0.5900 (≈ 59 %)
- MSE ≈ 1015.68
- RMSE ≈ 31.87

La Forêt Aléatoire surperforme nettement les modèles linéaires, mais reste en retrait par rapport à l’arbre de décision simple sur ce dataset particulier (paramètres par défaut).

### 4.5 Régression SVR

SVR avec noyau RBF, après **standardisation** des features (StandardScaler).

- R² ≈ 0.0899
- MSE ≈ 2238.45
- RMSE ≈ 47.31

Performance comparable à celle de la régression linéaire, indiquant que le modèle, dans sa configuration par défaut, ne parvient pas à exploiter efficacement les structures non linéaires présentes.

Un tuning des hyperparamètres (`C`, `gamma`, `epsilon`) serait nécessaire pour améliorer ce modèle.

### 4.6 Tableau Comparatif des Performances

| Modèle                   | R²       | MSE       | RMSE    | Performance          |
|--------------------------|----------|-----------|---------|----------------------|
| Régression Linéaire      | 0.0899   | 2238.45   | 47.31   | Très faible          |
| Régression Polynomiale   | 0.1706   | 2062.18   | 45.41   | Faible               |
| Arbre de Décision        | 0.7126   | 707.89    | 26.61   | Excellent            |
| Forêt Aléatoire          | 0.5900   | 1015.68   | 31.87   | Très bon             |
| SVR                      | 0.0899   | 2238.45   | 47.31   | Très faible          |

---

## 5. Analyse des Résultats et Recommandations

### Modèle Gagnant : Arbre de Décision

L’**Arbre de Décision** est le modèle le plus performant :

- R² ≈ 0.71
- RMSE ≈ 26.61

Il explique une grande partie de la variance du taux d’engagement tout en maintenant une erreur moyenne relativement faible.

### Interprétation

- Les relations entre les variables (`likes`, `comments`, `shares`, `reach`, etc.) et `engagement_rate` sont clairement **non linéaires**.
- Les modèles d’arbres sont adaptés à ces structures complexes et aux interactions entre les features.
- La Forêt Aléatoire est compétitive mais sous-performe l’arbre simple, probablement faute de tuning d’hyperparamètres.

### Recommandations pour Améliorer le Modèle

1. Optimiser l’Arbre de Décision :
   - Tuning de `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, etc.
2. Améliorer la Forêt Aléatoire :
   - Augmenter `n_estimators`,
   - Ajuster `max_depth`, `max_features`, `min_samples_leaf`, etc.
3. Améliorer le SVR :
   - Tuning de `C`, `gamma`, `epsilon` après standardisation.
4. Feature engineering avancé :
   - Ratios (`likes / reach`, `comments / reach`, `shares / reach`, etc.),
   - Catégorisation des heures/jours (créneaux horaires),
   - Interactions spécifiques entre variables d’engagement et catégories de contenu.
5. Validation plus robuste :
   - Utilisation de la **validation croisée** (k-fold),
   - Comparaison de modèles d’ensemble plus avancés (Gradient Boosting, XGBoost, LightGBM).

---

## 6. Conclusion

Ce projet montre comment appliquer un pipeline complet de **Data Science** à un cas réel de **marketing digital** (Instagram) :

- Prétraitement : extraction de features temporelles, encodage des variables catégorielles, nettoyage.
- EDA : compréhension des distributions, corrélations, échelles.
- Modélisation : comparaison de plusieurs algorithmes de régression.
- Résultat : l’**Arbre de Décision** est le meilleur modèle testé, avec un R² ≈ 0.71.

Même si la performance est déjà satisfaisante, des gains supplémentaires sont possibles via :

- le tuning d’hyperparamètres,
- un feature engineering plus poussé,
- l’utilisation de modèles d’ensemble plus puissants.

---

