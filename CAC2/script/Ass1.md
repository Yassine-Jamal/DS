# Analyse approfondie de la base de données "Wine Quality"
# Source: UCI Machine Learning Repository
# Référence: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009)
# Lien : https://archive.ics.uci.edu/dataset/186/wine+quality

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 1. Chargement des données
# ============================
red = pd.read_csv('winequality-red.csv', sep=';')
white = pd.read_csv('winequality-white.csv', sep=';')

# Affichage des dimensions des jeux de données
print(f"Nombre d'échantillons (vin rouge): {red.shape[0]}")
print(f"Nombre d'échantillons (vin blanc): {white.shape[0]}")

# ============================
# 2. Description des variables
# ============================
print("\nVariables disponibles :")
for col in red.columns:
    print(f"- {col}")

# ============================
# 3. Aperçu des données
# ============================
print("\nAperçu des premières lignes (vin rouge) :")
print(red.head())

# ============================
# 4. Statistiques descriptives
# ============================
print("\nStatistiques descriptives des vins rouges :")
print(red.describe())
print("\nStatistiques descriptives des vins blancs :")
print(white.describe())

# ============================
# 5. Visualisation de la distribution de la qualité
# ============================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.countplot(x='quality', data=red, palette='Reds')
plt.title("Distribution de la qualité - Vin rouge")
plt.xlabel("Score de qualité")
plt.ylabel("Nombre d'échantillons")

plt.subplot(1,2,2)
sns.countplot(x='quality', data=white, palette='Blues')
plt.title("Distribution de la qualité - Vin blanc")
plt.xlabel("Score de qualité")
plt.ylabel("Nombre d'échantillons")
plt.tight_layout()
plt.show()
# On constate que la plupart des vins ont des scores moyens : il y a peu de vins "excellents" ou "très mauvais".

# ============================
# 6. Analyse des corrélations
# ============================
plt.figure(figsize=(10,8))
sns.heatmap(red.corr(), cmap='Reds', annot=True, fmt=".2f")
plt.title("Matrice de corrélations (vin rouge)")
plt.show()

# Les variables comme l'alcool semblent avoir une corrélation positive avec la qualité du vin rouge.
# En revanche, la densité et l'acidité volatile montrent des corrélations négatives.

# ============================
# 7. Comparaison Rouge / Blanc sur quelques variables
# ============================
plt.figure(figsize=(10,6))
sns.boxplot(data=[red['alcohol'], white['alcohol']], notch=True)
plt.xticks([0,1], ['Rouge', 'Blanc'])
plt.title('Distribution de l\'alcool (%)')
plt.show()

# Les vins blancs ont en général un niveau d'alcool légèrement supérieur aux rouges.

# ============================
# 8. Relation entre alcool et qualité
# ============================
plt.figure(figsize=(10,5))
sns.boxplot(x='quality', y='alcohol', data=red, palette='Reds')
plt.title('Relation entre alcool et qualité - Vin rouge')
plt.xlabel('Qualité')
plt.ylabel('Alcool (%)')
plt.show()

# Plus le score de qualité est élevé, plus l'alcool tend à être élevé.

# ============================
# 9. Interprétation générale
# ============================
""" 
Commentaire :
- La qualité du vin dépend de multiples facteurs, parmi lesquels l'acidité volatile (corrélation souvent négative), le taux d'alcool (corrélation positive), et la densité.
- Les scores de qualité sont déséquilibrés (majorité de scores moyens).
- L'analyse met en avant l'intérêt de sélectionner judicieusement les variables pour bâtir un modèle prédictif robuste.
- Ce jeu de données illustre bien les défis classiques en data science appliquée à l'œnologie : multicolinéarité, classes déséquilibrées, importance de l'analyse exploratoire préalable.
"""

# ============================
# 10. Référence
# ============================
"""
Données issues de l’étude : 
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). 
Wine Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T
Licencié sous Creative Commons Attribution 4.0
"""
