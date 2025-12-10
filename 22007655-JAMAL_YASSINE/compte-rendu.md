# 🧮 PROJET DATA SCIENCE : RECONNAISSANCE DE CHIFFRES MANUSCRITS (DIGITS)

Ce document présente le cycle complet d’un mini-projet de Machine Learning appliqué au dataset **Digits** de Scikit-Learn, depuis le chargement des données jusqu’à l’audit de performance du modèle Random Forest.

---

## 1. Contexte et objectif

### Problème traité
Le problème consiste à reconnaître automatiquement des chiffres manuscrits à partir d’images 8x8 en niveaux de gris, chaque image représentant un chiffre entre 0 et 9.  
L’objectif est de construire un modèle de classification supervisée capable de prédire correctement le chiffre correspondant à une nouvelle image manuscrite. 

### Données utilisées
Le dataset **Digits** contient :  
- 1 797 observations, chacune correspondant à une image 8x8 (soit 64 pixels) aplatie en vecteur de dimension 64.   
- Une cible `target` prenant des valeurs de 0 à 9, représentant la classe du chiffre manuscrit. 

Les variables explicatives sont des intensités de pixels (valeurs entières entre 0 et 16), et la variable cible est un entier indiquant le chiffre manuscrit.

---

## 2. Laboratoire Python (script Colab)

### Bibliothèques et chargement du dataset
Le script importe les principales bibliothèques de data science : NumPy, pandas, Matplotlib, Seaborn et plusieurs modules de Scikit-Learn (chargement du dataset, split, imputation, RandomForest, métriques).  
Le dataset **Digits** est chargé via `load_digits()`, puis transformé en `DataFrame` pour les features (colonnes `pixel_0` à `pixel_63`) et en `Series` pour la cible `target`. 

### Structure générale du code
Le notebook suit une structure pédagogique claire :  
1. Importation des bibliothèques. 
2. Chargement et inspection du dataset.  
3. Simulation de données manquantes.   
4. Nettoyage / imputation.   
5. Analyse exploratoire (statistiques + visualisations).  
6. Découpage Train / Test.  
7. Entraînement d’un modèle Random Forest.   
8. Évaluation via accuracy, rapport de classification et matrice de confusion. 

---

## 3. Nettoyage des données (Data Wrangling)

### Simulation des valeurs manquantes
Pour rendre l’exercice plus réaliste, des valeurs manquantes artificielles sont introduites :  
- Pour chaque pixel (chaque colonne de feature), 5 % des lignes sont remplacées par `NaN`.   
- Le nombre total de valeurs manquantes générées atteint 5 760, ce qui correspond à une perturbation significative sur l’ensemble des features. 

Ce choix permet de tester une vraie étape de **gestion des données incomplètes**, fréquente en production. 

### Imputation et reconstruction du jeu propre
Le nettoyage est réalisé en deux temps :  
- Séparation des données en `X` (toutes les colonnes de pixels) et `y` (colonne `target`).   
- Application d’un `SimpleImputer(strategy="mean")` sur `X` pour remplacer chaque `NaN` par la moyenne de la colonne correspondante, puis reconstruction d’un `DataFrame` `X_clean`. 

Après imputation, le script vérifie qu’il ne reste plus aucune valeur manquante, ce qui garantit que les algorithmes de Machine Learning pourront fonctionner correctement. 

---

## 4. Analyse exploratoire (EDA)

### Statistiques descriptives des pixels
Le script affiche `.describe()` pour les 10 premiers pixels (`pixel_0` à `pixel_9`).   
On observe notamment :  
- Des minimums à 0 et des maximums à 16, cohérents avec l’échelle des intensités de gris du dataset Digits.   
- Des distributions où la médiane et la moyenne peuvent diverger, indiquant parfois des distributions asymétriques selon le pixel. 

### Visualisation des images et distributions
Plusieurs visualisations complètent le profilage :  
- Un panel de 10 images 8x8 est affiché avec leur label réel, ce qui donne une intuition qualitative de la difficulté du problème.   
- Une distribution d’un pixel choisi (`pixel_20`) est tracée en fonction de la classe `target`, illustrant comment une même zone de l’image peut porter une information discriminante selon le chiffre. 

### Corrélations entre features
Une matrice de corrélation est calculée sur les 20 premiers pixels puis visualisée par une heatmap Seaborn.   
Cette visualisation permet d’identifier des groupes de pixels fortement corrélés, souvent voisins dans l’image, reflétant la structure spatiale des chiffres manuscrits. 

---

## 5. Méthodologie expérimentale (Train / Test Split)

### Stratégie de découpage
La séparation des données se fait via `train_test_split` avec les paramètres :  
- `test_size=0.2` pour garder 20 % des données pour le test (environ 360 échantillons).  
- `random_state=42` pour la reproductibilité des résultats.  
- `stratify=y` pour conserver la même proportion de chaque chiffre dans les ensembles d’entraînement et de test. 

Cette stratégie garantit :  
- Un apprentissage sur une base suffisamment riche (80 % des données). 
- Une évaluation fiable sur un échantillon représentatif et équilibré des classes. 

---

## 6. Modèle de Machine Learning : Random Forest

### Choix de l’algorithme
Le modèle utilisé est un **RandomForestClassifier**, bien adapté aux problèmes de classification multi-classes comme Digits (10 classes de 0 à 9).  
Les hyperparamètres principaux sont :  
- `n_estimators=200` (nombre d’arbres dans la forêt).   
- `max_depth=None` (profondeur non limitée, laissée à l’algorithme). 
- `random_state=42` et `n_jobs=-1` pour la reproductibilité et l’exploitation de tous les cœurs CPU. 

### Entraînement
Le modèle est entraîné sur `X_train` et `y_train` via `model.fit`.   
Cette étape apprend les motifs entre combinaisons de pixels et classes de chiffres manuscrits sur l’ensemble d’entraînement nettoyé. 

---

## 7. Évaluation du modèle

### Accuracy globale
L’accuracy est calculée sur le jeu de test :  
- Le score obtenu est d’environ **plus de 95 %** (précision globale très élevée sur la classification des chiffres).  
- Cela montre que le Random Forest capture efficacement la structure des chiffres manuscrits dans ce dataset. 

### Rapport de classification
Le rapport détaillé (`classification_report`) fournit, pour chaque classe 0–9 :  
- La précision (precision), le rappel (recall) et le F1-score.  
- Des scores globalement élevés et homogènes, ce qui indique que le modèle ne se contente pas de bien prédire quelques chiffres seulement. 

### Matrice de confusion
Une matrice de confusion est tracée via `sns.heatmap`, avec les chiffres réels en ordonnée et les prédictions en abscisse. [file:2]  
Elle montre que :  
- La majorité des prédictions sont sur la diagonale, signe d’une bonne classification.  
- Quelques confusions subsistent entre certains chiffres visuellement proches (par exemple 3/5 ou 4/9), ce qui donne des pistes pour de futures améliorations. 

---

## 8. Lecture critique et axes d’amélioration

### Points forts de la démarche
- Pipeline complet : du chargement des données au reporting final, avec une structure claire et pédagogique.  
- Gestion explicite des valeurs manquantes et vérification post-imputation. 
- Utilisation de visualisations pertinentes (images, distributions, corrélations, matrice de confusion). 
- Modèle robuste (Random Forest) capable de gérer des features corrélées et d’obtenir une haute accuracy sur un problème multi-classes.

### Limites et pistes d’extension
- L’imputation est faite sur l’ensemble des données avant le split, ce qui serait à corriger dans un pipeline industriel (risque de **data leakage**).   
- Une exploration d’autres modèles (par exemple SVM ou réseaux de neurones) ou un tuning plus systématique des hyperparamètres pourrait encore améliorer la performance. 

---

## 9. Conclusion

Ce projet illustre un **cycle de vie complet** en Data Science sur un problème de vision simple :  
- Préparation et nettoyage d’un dataset d’images vectorisées.   
- Analyse exploratoire pour comprendre les distributions de pixels et leurs liens avec les classes.   
- Modélisation avec un algorithme robuste (Random Forest) et évaluation fine via plusieurs métriques et visualisations. 

L’ensemble du travail montre comment transformer un notebook Colab en un véritable **projet structuré** de reconnaissance de chiffres manuscrits. 
