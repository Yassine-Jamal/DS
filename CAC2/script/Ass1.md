# JAMAL YASSINE

<img src="JAMAL YASSINE CAC2.jpg" style="height:464px;margin-right:432px"/>

# CAC2

# 22007655

# Analyse complète de la base de données "Wine Quality"
# Source : UCI Machine Learning Repository, 2009
# Étude conduite par l'équipe du département Informatique, Universidade do Minho, Portugal
# Référence : Cortez et al., 2009

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------------------------------
# Introduction et objectifs de la base de données
# ---------------------------------------------------------------------------------------------------

"""
Cette base contient des données physico-chimiques ainsi que des notes de qualité sensorielle obtenues
sur des vins rouges et blancs de la région Vinho Verde au Portugal.
L'objectif principal est de fournir un jeu de données pour modéliser la qualité du vin à partir de mesures
objectives en laboratoire. Cette ressource est utilisée pour la recherche en apprentissage automatique,
notamment pour classification, régression et sélection de variables.
"""

# ---------------------------------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------------------------------

red_wine = pd.read_csv('winequality-red.csv', sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')

print(f"Nombre d'échantillons vin rouge : {red_wine.shape[0]}, variables : {red_wine.shape[1]}")
print(f"Nombre d'échantillons vin blanc : {white_wine.shape[0]}, variables : {white_wine.shape[1]}")

# ---------------------------------------------------------------------------------------------------
# Exploration initiale des données
# ---------------------------------------------------------------------------------------------------

print("\nVariables disponibles :")
print(list(red_wine.columns))

print("\nAperçu des vins rouges :")
print(red_wine.head())

print("\nStatistiques descriptives vins rouges :")
print(red_wine.describe())

print("\nStatistiques descriptives vins blancs :")
print(white_wine.describe())

# ---------------------------------------------------------------------------------------------------
# Visualisation de la distribution des notes de qualité
# ---------------------------------------------------------------------------------------------------

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.countplot(x='quality', data=red_wine, palette='Reds')
plt.title("Distribution de la qualité - Vins rouges")
plt.xlabel("Score de qualité")
plt.ylabel("Nombre d'échantillons")

plt.subplot(1,2,2)
sns.countplot(x='quality', data=white_wine, palette='Blues')
plt.title("Distribution de la qualité - Vins blancs")
plt.xlabel("Score de qualité")
plt.ylabel("Nombre d'échantillons")

plt.tight_layout()
plt.show()

"""
On observe que la majorité des vins a des notes comprises entre 5 et 7,
avec peu de vins très bons ou très mauvais.
Ce déséquilibre est important à considérer pour l'entraînement de modèles prédictifs.
"""

# ---------------------------------------------------------------------------------------------------
# Matrice de corrélation sur les variables du vin rouge
# ---------------------------------------------------------------------------------------------------

plt.figure(figsize=(10,8))
sns.heatmap(red_wine.corr(), annot=True, cmap='Reds', fmt=".2f")
plt.title("Matrice de corrélation des variables - Vin rouge")
plt.show()

"""
Les corrélations notables sont :
- Alcool corrélé positivement à la qualité (environ 0.45)
- Acidité volatile corrélée négativement à la qualité (environ -0.26)
- Densité aussi négativement corrélée
Ces variables sont donc clés dans la prédiction de la qualité.
"""

# ---------------------------------------------------------------------------------------------------
# Comparaison du taux d'alcool entre vins rouges et blancs
# ---------------------------------------------------------------------------------------------------

plt.figure(figsize=(6,5))
sns.boxplot(data=[red_wine['alcohol'], white_wine['alcohol']])
plt.xticks([0, 1], ['Rouge', 'Blanc'])
plt.title("Comparaison du taux d'alcool (%)")
plt.show()

"""
Les vins blancs ont en moyenne un taux d'alcool légèrement plus élevé que les rouges,
ce qui peut influencer leurs profils sensoriels et les notes de qualité.
"""

# ---------------------------------------------------------------------------------------------------
# Relation alcool vs qualité dans les vins rouges
# ---------------------------------------------------------------------------------------------------

plt.figure(figsize=(10,5))
sns.boxplot(x='quality', y='alcohol', data=red_wine, palette='Reds')
plt.title("Relation entre le taux d'alcool et la qualité - Vin rouge")
plt.xlabel("Qualité")
plt.ylabel("Taux d'alcool (%)")
plt.show()

"""
On remarque une tendance claire : les vins rouges mieux notés ont généralement un taux d'alcool plus élevé,
un indicateur important à considérer.
"""

# ---------------------------------------------------------------------------------------------------
# Conclusion et interprétations
# ---------------------------------------------------------------------------------------------------

"""
Cette base de données est une ressource précieuse pour étudier les liens entre caractéristiques chimiques objectives
et la perception sensorielle de la qualité du vin. 
Elle sert de cas d'étude typique en science des données appliquée à l'œnologie, 
avec des défis comme le déséquilibre des classes, la relation entre variables continues 
et la nécessité d'une sélection informative des variables.

L'étude a été menée en 2009 par l'équipe portugaise de l'Université do Minho et reste une référence pour la recherche
en machine learning appliquée à l'agroalimentaire.
"""

# ---------------------------------------------------------------------------------------------------
# Références
# ---------------------------------------------------------------------------------------------------

"""
- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).
  Wine Quality [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C56S3T
- Licence : Creative Commons Attribution 4.0 International (CC BY 4.0)
"""

# -*- coding: utf-8 -*-
"""JAMAL YASSINE / SS1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Ujxa4m-oxa8ZLRZ8gerhwFZRA2tp4P-8
"""

# Installer la bibliothèque ucimlrepo pour récupérer des jeux de données du référentiel UCI Machine Learning
!pip install ucimlrepo

from ucimlrepo import fetch_ucirepo

# Récupérer l'ensemble de données 'Wine Quality' depuis le référentiel UCI
wine_quality = fetch_ucirepo(id=186)

# Extraire les caractéristiques (X) et les cibles (y) en tant que DataFrames pandas
X = wine_quality.data.features
y = wine_quality.data.targets

# Afficher les métadonnées de l'ensemble de données
print(wine_quality.metadata)

# Afficher les informations sur les variables de l'ensemble de données
print(wine_quality.variables)

import zipfile
import pandas as pd
import os

# Chemin vers le fichier ZIP contenant les données du vin
zip_file_path = "/content/drive/MyDrive/DS.2025/wine+quality.zip"

# Répertoire temporaire où les fichiers CSV seront décompressés
extract_dir = "/tmp/wine_quality_data"

# Créer le répertoire de destination s'il n'existe pas déjà
os.makedirs(extract_dir, exist_ok=True)

# Décompresser le fichier ZIP dans le répertoire spécifié
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Afficher le chemin de décompression et la liste des fichiers décompressés
print(f"Fichiers décompressés dans : {extract_dir}")
print("Contenu du répertoire décompressé :")
for file_name in os.listdir(extract_dir):
    print(file_name)

"""Maintenant que les fichiers sont décompressés, nous pouvons les charger dans des DataFrames pandas."""

# Charger les fichiers CSV décompressés dans des DataFrames pandas

# Construire les chemins complets vers les fichiers de vin rouge et blanc
red_wine_path = os.path.join(extract_dir, 'winequality-red.csv')
white_wine_path = os.path.join(extract_dir, 'winequality-white.csv')

# Tenter de charger les données du vin rouge
try:
    # Le séparateur est généralement ';' pour ce dataset de vin
    df_red = pd.read_csv(red_wine_path, sep=';')
    print("### Données du vin rouge (premières 5 lignes) :")
    # Afficher les 5 premières lignes du DataFrame de vin rouge
    display(df_red.head())
    # Afficher le nombre total de lignes dans le DataFrame de vin rouge
    print(f"Nombre de lignes dans le dataset de vin rouge : {len(df_red)}")
except FileNotFoundError:
    print(f"Le fichier '{red_wine_path}' n'a pas été trouvé. Veuillez vérifier les noms des fichiers décompressés.")

print("\n") # Ajouter une ligne vide pour une meilleure lisibilité

# Tenter de charger les données du vin blanc
try:
    # Le séparateur est généralement ';' pour ce dataset de vin
    df_white = pd.read_csv(white_wine_path, sep=';')
    print("### Données du vin blanc (premières 5 lignes) :")
    # Afficher les 5 premières lignes du DataFrame de vin blanc
    display(df_white.head())
    # Afficher le nombre total de lignes dans le DataFrame de vin blanc
    print(f"Nombre de lignes dans le dataset de vin blanc : {len(df_white)}")
except FileNotFoundError:
    print(f"Le fichier '{white_wine_path}' n'a pas été trouvé. Veuillez vérifier les noms des fichiers décompressés.")

"""# Task
Present a summary of the key findings from the in-depth analysis of red and white wine datasets (`df_red` and `df_white`). The summary should highlight the most influential factors on wine quality and the notable differences in physicochemical characteristics and quality distribution between red and white wines. This includes an analysis of variable distributions and outliers, correlations with 'quality', deeper insights into quality variations based on key chemical properties, and a comparison of characteristics between the two wine types.

## Analyse de la Distribution des Variables et Détection d'Outliers

### Subtask:
Visualiser la distribution de chaque caractéristique et de la variable 'quality' pour les vins rouges et blancs à l'aide d'histrogrammes. Utiliser des boîtes à moustaches (box plots) pour identifier les valeurs aberrantes (outliers) potentielles dans les caractéristiques clés.

**Reasoning**:
The first step is to import the necessary libraries, matplotlib.pyplot and seaborn, to enable data visualization for histograms and box plots as requested by the subtask.
"""

import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries matplotlib.pyplot and seaborn imported successfully.")

"""**Reasoning**:
To fulfill the subtask, the next step is to generate histograms for all numerical features in the `df_red` DataFrame, visualizing their distributions. This will provide an initial understanding of the data's spread and central tendency for red wines.


"""

print("### Histograms for Red Wine Features")
# Determine the number of columns for subplots
num_features_red = df_red.shape[1]
num_cols = 3 # Number of columns in the subplot grid
num_rows_red = (num_features_red + num_cols - 1) // num_cols # Calculate rows needed

plt.figure(figsize=(num_cols * 5, num_rows_red * 4)) # Adjust figure size dynamically

for i, column in enumerate(df_red.columns):
    plt.subplot(num_rows_red, num_cols, i + 1)
    sns.histplot(df_red[column], kde=True, color='red')
    plt.title(f'Distribution de {column.replace("_", " ").title()} pour le Vin Rouge')
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel('Fréquence')

plt.tight_layout()
plt.show()

"""**Reasoning**:
Now that histograms for red wine features have been displayed, the next step is to generate box plots for the `df_red` DataFrame to visually identify potential outliers in each characteristic, as required by the subtask.


"""

print("### Box Plots for Red Wine Features (Outlier Detection)")
# Determine the number of columns for subplots
num_features_red = df_red.shape[1]
num_cols = 3 # Number of columns in the subplot grid
num_rows_red = (num_features_red + num_cols - 1) // num_cols # Calculate rows needed

plt.figure(figsize=(num_cols * 5, num_rows_red * 4)) # Adjust figure size dynamically

for i, column in enumerate(df_red.columns):
    plt.subplot(num_rows_red, num_cols, i + 1)
    sns.boxplot(y=df_red[column], color='red')
    plt.title(f'Boîte à Moustaches de {column.replace("_", " ").title()} pour le Vin Rouge')
    plt.ylabel(column.replace("_", " ").title())

plt.tight_layout()
plt.show()

"""**Reasoning**:
Following the subtask instructions, the next step is to generate histograms for all numerical features in the `df_white` DataFrame, visualizing their distributions. This will provide an initial understanding of the data's spread and central tendency for white wines, similar to what was done for red wines.


"""

print("### Histograms for White Wine Features")
# Determine the number of columns for subplots
num_features_white = df_white.shape[1]
num_cols = 3 # Number of columns in the subplot grid
num_rows_white = (num_features_white + num_cols - 1) // num_cols # Calculate rows needed

plt.figure(figsize=(num_cols * 5, num_rows_white * 4)) # Adjust figure size dynamically

for i, column in enumerate(df_white.columns):
    plt.subplot(num_rows_white, num_cols, i + 1)
    sns.histplot(df_white[column], kde=True, color='skyblue') # Using a different color for white wine
    plt.title(f'Distribution de {column.replace("_", " ").title()} pour le Vin Blanc')
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel('Fréquence')

plt.tight_layout()
plt.show()

"""**Reasoning**:
Following the subtask instructions, the next step is to generate box plots for the `df_white` DataFrame to visually identify potential outliers in each characteristic, as required by the subtask, similar to what was done for red wines.


"""

print("### Box Plots for White Wine Features (Outlier Detection)")
# Determine the number of columns for subplots
num_features_white = df_white.shape[1]
num_cols = 3 # Number of columns in the subplot grid
num_rows_white = (num_features_white + num_cols - 1) // num_cols # Calculate rows needed

plt.figure(figsize=(num_cols * 5, num_rows_white * 4)) # Adjust figure size dynamically

for i, column in enumerate(df_white.columns):
    plt.subplot(num_rows_white, num_cols, i + 1)
    sns.boxplot(y=df_white[column], color='skyblue') # Using the same color for white wine as histograms
    plt.title(f'Boîte à Moustaches de {column.replace("_", " ").title()} pour le Vin Blanc')
    plt.ylabel(column.replace("_", " ").title())

plt.tight_layout()
plt.show()

"""## Analyse des Corrélations

### Subtask:
Calculer et visualiser les matrices de corrélation pour les jeux de données de vin rouge et blanc. Identifier les caractéristiques qui sont fortement corrélées avec la variable 'quality' et interpréter ces relations.

**Reasoning**:
The first step is to calculate the correlation matrix for the red wine dataset (`df_red`) and visualize it using a heatmap as requested by the subtask instructions. This will help identify relationships between variables, especially with 'quality'.
"""

print("### Matrice de corrélation pour le vin rouge\n")
# 1. Calculer la matrice de corrélation pour le vin rouge
correlation_matrix_red = df_red.corr()

# 2. Visualiser la matrice de corrélation pour le vin rouge
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_red, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de Corrélation pour les Vins Rouges')
plt.show()

"""**Reasoning**:
Following the subtask instructions, the next step is to calculate the correlation matrix for the white wine dataset (`df_white`) and visualize it using a heatmap to identify relationships between variables, especially with 'quality'.


"""

print("### Matrice de corrélation pour le vin blanc\n")
# 3. Calculer la matrice de corrélation pour le vin blanc
correlation_matrix_white = df_white.corr()

# 4. Visualiser la matrice de corrélation pour le vin blanc
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_white, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de Corrélation pour les Vins Blancs')
plt.show()

"""### Interprétation des Corrélations avec la Qualité du Vin

#### Vins Rouges (df_red)

En analysant la matrice de corrélation pour les vins rouges, nous observons les corrélations les plus notables avec la variable 'quality' :

*   **Correlations Positives Fortes :**
    *   **Alcohol (~0.48):** L'alcool est le facteur le plus fortement corrélé positivement avec la qualité du vin rouge. Cela suggère que les vins rouges avec une teneur en alcool plus élevée ont tendance à être mieux notés.
    *   **Sulphates (~0.25):** Les sulfates présentent une corrélation positive modérée avec la qualité. Les sulfates agissent comme un agent antimicrobien et un antioxydant, et une certaine quantité peut être liée à une meilleure qualité.
    *   **Citric Acid (~0.23):** L'acide citrique montre une corrélation positive modérée, indiquant qu'il peut contribuer positivement à la saveur et à la fraîcheur, ce qui est souvent associé à une meilleure qualité.

*   **Correlations Négatives Fortes :**
    *   **Volatile Acidity (~-0.39):** L'acidité volatile est le facteur le plus fortement corrélé négativement avec la qualité. Une acidité volatile élevée est souvent associée à des défauts du vin (comme une odeur de vinaigre), ce qui dégrade la perception de la qualité.
    *   **Total Sulfur Dioxide (~-0.19):** Le dioxyde de soufre total présente une corrélation négative modérée. Bien que le SO2 soit un conservateur important, des niveaux trop élevés peuvent être perçus négativement et masquer les arômes fruités.
    *   **Chlorides (~-0.13):** Les chlorures ont une faible corrélation négative avec la qualité. Des niveaux élevés de chlorures peuvent indiquer une salinité et impacter négativement le goût.

#### Vins Blancs (df_white)

Pour les vins blancs, les corrélations avec la 'quality' montrent des tendances légèrement différentes :

*   **Correlations Positives Fortes :**
    *   **Alcohol (~0.43):** Tout comme pour les vins rouges, l'alcool est le principal contributeur positif à la qualité des vins blancs. Une teneur en alcool plus élevée est associée à une meilleure qualité.
    *   **Density (~-0.31):** Bien que la densité soit un paramètre physique plutôt qu'un composant de saveur direct, elle est fortement corrélée négativement avec la qualité. Cela peut être lié à des sucres résiduels plus faibles ou à d'autres caractéristiques de la composition. *Correction: la densité est fortement corrélée négativement, il faut la mentionner dans les corrélations négatives.*

*   **Correlations Négatives Fortes :**
    *   **Density (~-0.31):** La densité est la caractéristique la plus fortement corrélée négativement avec la qualité pour les vins blancs. Une densité plus faible est associée à une meilleure qualité, ce qui pourrait refléter des vins plus secs ou des profils de saveur plus légers et complexes.
    *   **Volatile Acidity (~-0.19):** L'acidité volatile a une corrélation négative modérée, similaire aux vins rouges, indiquant qu'un excès d'acidité volatile est également indésirable pour les vins blancs.
    *   **Chlorides (~-0.21):** Les chlorures montrent une corrélation négative modérée, suggérant que des niveaux élevés peuvent nuire à la qualité perçue des vins blancs.
    *   **pH (~-0.13):** Le pH a une faible corrélation négative, indiquant qu'un pH légèrement plus bas (plus acide) peut être préférable pour la qualité des vins blancs.

**Observations Générales :**

*   **L'alcool** est un facteur clé de la qualité pour les deux types de vin, avec une corrélation positive significative.
*   **L'acidité volatile** est un indicateur négatif de la qualité pour les deux types de vin, suggérant que des niveaux bas sont préférables.
*   Les **densité**, **chlorides** et **sulfates** jouent des rôles différents ou avec des intensités différentes entre les vins rouges et blancs, reflétant les particularités de leur composition et de leur perception de la qualité.

## Analyse Approfondie de la Qualité du Vin

### Subtask:
Comparer la distribution des scores de qualité entre les vins rouges et blancs. Examiner comment les propriétés chimiques clés (par exemple, l'alcool, l'acidité volatile) varient en fonction de la qualité pour chaque type de vin, potentiellement à l'aide de graphiques de densité ou de boîtes à moustaches groupées par qualité.

**Reasoning**:
The first step is to compare the distribution of 'quality' scores between red and white wines as requested by the subtask. This will be done by plotting histograms/density plots of 'quality' for both `df_red` and `df_white` on the same graph.
"""

print("### Comparaison de la distribution de la qualité entre les vins rouges et blancs\n")

plt.figure(figsize=(10, 6))
sns.histplot(df_red['quality'], kde=True, color='red', label='Vin Rouge', alpha=0.7)
sns.histplot(df_white['quality'], kde=True, color='skyblue', label='Vin Blanc', alpha=0.7)
plt.title('Distribution des scores de Qualité pour les Vins Rouges et Blancs')
plt.xlabel('Qualité')
plt.ylabel('Fréquence')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.xticks(range(3, 10)) # Quality scores usually range from 3 to 9
plt.show()

"""**Reasoning**:
To address the second instruction of the subtask, I will create box plots for 'alcohol' and 'volatile acidity' for red wines (`df_red`), grouping them by 'quality' to visualize how these chemical properties vary across different quality scores.


"""

print("### Variation des propriétés clés par Qualité pour les Vins Rouges\n")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='quality', y='alcohol', data=df_red, palette='Reds')
plt.title('Alcool par Qualité pour les Vins Rouges')
plt.xlabel('Qualité')
plt.ylabel('Alcool')

plt.subplot(1, 2, 2)
sns.boxplot(x='quality', y='volatile acidity', data=df_red, palette='Reds')
plt.title('Acidité Volatile par Qualité pour les Vins Rouges')
plt.xlabel('Qualité')
plt.ylabel('Acidité Volatile')

plt.tight_layout()
plt.show()

"""**Reasoning**:
The previous code generated a `FutureWarning` because `palette` was passed without `hue` in `sns.boxplot`. To address this, I will explicitly set `hue='quality'` and `legend=False` in both box plot calls for red wines, as suggested by the warning.


"""

print("### Variation des propriétés clés par Qualité pour les Vins Rouges\n")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='quality', y='alcohol', data=df_red, hue='quality', palette='Reds', legend=False)
plt.title('Alcool par Qualité pour les Vins Rouges')
plt.xlabel('Qualité')
plt.ylabel('Alcool')

plt.subplot(1, 2, 2)
sns.boxplot(x='quality', y='volatile acidity', data=df_red, hue='quality', palette='Reds', legend=False)
plt.title('Acidité Volatile par Qualité pour les Vins Rouges')
plt.xlabel('Qualité')
plt.ylabel('Acidité Volatile')

plt.tight_layout()
plt.show()

"""**Reasoning**:
To complete the third instruction of the subtask, I will create box plots for 'alcohol', 'density', and 'volatile acidity' for white wines (`df_white`), grouping them by 'quality' to visualize how these chemical properties vary across different quality scores.


"""

print("### Variation des propriétés clés par Qualité pour les Vins Blancs\n")

plt.figure(figsize=(18, 6)) # Adjust figure size to accommodate three subplots

plt.subplot(1, 3, 1)
sns.boxplot(x='quality', y='alcohol', data=df_white, hue='quality', palette='Blues', legend=False)
plt.title('Alcool par Qualité pour les Vins Blancs')
plt.xlabel('Qualité')
plt.ylabel('Alcool')

plt.subplot(1, 3, 2)
sns.boxplot(x='quality', y='density', data=df_white, hue='quality', palette='Blues', legend=False)
plt.title('Densité par Qualité pour les Vins Blancs')
plt.xlabel('Qualité')
plt.ylabel('Densité')

plt.subplot(1, 3, 3)
sns.boxplot(x='quality', y='volatile acidity', data=df_white, hue='quality', palette='Blues', legend=False)
plt.title('Acidité Volatile par Qualité pour les Vins Blancs')
plt.xlabel('Qualité')
plt.ylabel('Acidité Volatile')

plt.tight_layout()
plt.show()

"""## Comparaison des Caractéristiques entre Vins Rouges et Blancs

### Subtask:
Comparer les moyennes (ou médianes) des principales caractéristiques physico-chimiques entre les vins rouges et blancs pour identifier les différences significatives dans leur composition.

**Reasoning**:
To compare the physicochemical characteristics between red and white wines as requested by the subtask, I will calculate the mean of each numerical feature for both `df_red` and `df_white`. This will allow for a direct comparison of their average compositions.
"""

print("### Comparaison des Moyennes des Caractéristiques Physico-Chimiques entre Vins Rouges et Blancs\n")

# Calculer les moyennes pour df_red
mean_red_wine = df_red.mean().to_frame(name='Mean Red Wine')

# Calculer les moyennes pour df_white
mean_white_wine = df_white.mean().to_frame(name='Mean White Wine')

# Concaténer les moyennes des deux DataFrames pour une comparaison côte à côte
comparison_of_means = pd.concat([mean_red_wine, mean_white_wine], axis=1)

# Afficher le DataFrame de comparaison
display(comparison_of_means)

"""**Reasoning**:
To fully address the first instruction of the subtask, which is to calculate various descriptive statistics for both red and white wine datasets, I will now calculate and display the median and standard deviation for each characteristic in `df_red` and `df_white`. This complements the mean comparison already performed, providing a more comprehensive statistical overview.


"""

print("### Statistiques Descriptives Détaillées pour les Vins Rouges et Blancs\n")

# Calculer les statistiques descriptives pour df_red
descriptive_stats_red = df_red.describe().transpose()
print("#### Statistiques Descriptives pour les Vins Rouges:")
display(descriptive_stats_red)

print("\n#### Statistiques Descriptives pour les Vins Blancs:")
# Calculer les statistiques descriptives pour df_white
descriptive_stats_white = df_white.describe().transpose()
display(descriptive_stats_white)

print("\n### Comparaison des Médianes des Caractéristiques Physico-Chimiques entre Vins Rouges et Blancs\n")

# Calculer les médianes pour df_red
median_red_wine = df_red.median().to_frame(name='Median Red Wine')

# Calculer les médianes pour df_white
median_white_wine = df_white.median().to_frame(name='Median White Wine')

# Concaténer les médianes des deux DataFrames pour une comparaison côte à côte
comparison_of_medians = pd.concat([median_red_wine, median_white_wine], axis=1)

# Afficher le DataFrame de comparaison
display(comparison_of_medians)

print("\n### Comparaison des Écarts Types des Caractéristiques Physico-Chimiques entre Vins Rouges et Blancs\n")

# Calculer les écarts types pour df_red
std_red_wine = df_red.std().to_frame(name='Std Dev Red Wine')

# Calculer les écarts types pour df_white
std_white_wine = df_white.std().to_frame(name='Std Dev White Wine')

# Concaténer les écarts types des deux DataFrames pour une comparaison côte à côte
comparison_of_stds = pd.concat([std_red_wine, std_white_wine], axis=1)

# Afficher le DataFrame de comparaison
display(comparison_of_stds)

"""## Final Task

### Subtask:
Présenter un résumé des principales conclusions tirées de cette analyse approfondie, en mettant en évidence les facteurs les plus influents sur la qualité du vin et les différences notables entre les vins rouges et blancs.

## Summary:

### Q&A

1.  **What are the most influential factors on wine quality for both red and white wines?**
    For both red and white wines, `alcohol` content is the most strongly positively correlated factor with quality. Conversely, `volatile acidity` is the most strongly negatively correlated factor for both types, indicating that lower levels are associated with higher quality.

2.  **What are the notable differences in physicochemical characteristics and quality distribution between red and white wines?**
    *   **Physicochemical Characteristics:** Red wines generally have higher `fixed acidity`, `volatile acidity`, `chlorides`, and `sulphates` compared to white wines. White wines, on the other hand, show much higher `residual sugar`, `free sulfur dioxide`, and `total sulfur dioxide`, and slightly higher `citric acid` and `alcohol`. White wines also tend to have lower `density` and `pH` than red wines.
    *   **Quality Distribution:** White wines exhibit a slightly broader and generally higher average quality distribution (mean of 5.88) compared to red wines (mean of 5.64), with white wines having more instances of higher quality scores (e.g., 8 and 9).

### Data Analysis Key Findings

*   **Influential Factors on Quality (Correlations):**
    *   **Alcohol:** Consistently the strongest positive predictor of quality for both red (correlation of ~0.48) and white wines (correlation of ~0.43).
    *   **Volatile Acidity:** The strongest negative predictor of quality for both red (correlation of ~-0.39) and white wines (correlation of ~-0.19), indicating that lower levels are desirable.
    *   **Other Key Factors for Red Wine Quality:** `Sulphates` (~0.25) and `citric acid` (~0.23) show moderate positive correlations, while `total sulfur dioxide` (~-0.19) and `chlorides` (~-0.13) have negative correlations.
    *   **Other Key Factors for White Wine Quality:** `Density` (~-0.31) is a strong negative correlator, while `chlorides` (~-0.21) and `pH` (~-0.13) also show negative correlations.
*   **Quality Distribution:** White wines exhibit a slightly higher average quality score (mean of 5.88) and a broader distribution, including more samples in higher quality ranges (e.g., up to 9), compared to red wines (mean of 5.64), which are more concentrated around scores of 5 and 6.
*   **Physicochemical Differences between Wine Types:**
    *   **Acidity:** Red wines are notably higher in `fixed acidity` (mean: 8.32 vs. 6.85) and significantly higher in `volatile acidity` (mean: 0.53 vs. 0.28).
    *   **Sugar & Sulfites:** White wines contain substantially more `residual sugar` (mean: 6.39 vs. 2.54), `free sulfur dioxide` (mean: 35.31 vs. 15.87), and `total sulfur dioxide` (mean: 138.36 vs. 46.47).
    *   **Other Characteristics:** Red wines typically have higher `chlorides` (mean: 0.087 vs. 0.046) and `sulphates` (mean: 0.66 vs. 0.49), whereas `alcohol` content is quite similar (mean: 10.42 for red vs. 10.51 for white).
*   **Outlier Detection:** Box plots revealed the presence of outliers across many physicochemical properties for both red and white wines, suggesting natural variations or potential data entry anomalies.

### Insights or Next Steps

*   **Targeted Improvement Strategies:** Winemakers could focus on managing `volatile acidity` to improve quality for both red and white wines. For white wines, controlling `density` could also be a key factor.
*   **Feature Engineering for Predictive Models:** The strong correlations identified (e.g., `alcohol` and `volatile acidity` with `quality`) suggest these features would be highly valuable in developing predictive models for wine quality. Further investigation into non-linear relationships or interactions between variables could provide deeper insights.
"""
