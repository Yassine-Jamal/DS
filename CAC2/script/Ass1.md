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
