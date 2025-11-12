# Wine Quality Analysis
# Source: UCI Machine Learning Repository
# Dataset: https://archive.ics.uci.edu/dataset/186/wine+quality
# Reference: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality [Dataset]. UCI Machine Learning Repository.

import pandas as pd
import matplotlib.pyplot as plt

# --- Dataset Info ---
"""
- Étude réalisée par le département Informatique, Universidade do Minho, Portugal.
- But : prédire la qualité du vin (score 0-10) à partir des propriétés physico-chimiques (11 variables).
- Les données contiennent deux sous-groupes : vins rouges et vins blancs.
- Utilisations : analyse exploratoire, classification, régression, sélection de variables, détection d'outliers.
"""

# --- Chargement des données ---
red = pd.read_csv('winequality-red.csv', sep=';')
white = pd.read_csv('winequality-white.csv', sep=';')

# --- Vue rapide ---
print("Red wine shape:", red.shape)
print("White wine shape:", white.shape)
print("Red wine variables:", list(red.columns))
print("White wine variables:", list(white.columns))

# --- Statistiques descriptives ---
print("Red wine descriptive statistics:")
print(red.describe())
print("White wine descriptive statistics:")
print(white.describe())

# --- Distribution des scores de qualité ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
red['quality'].hist(color='red', bins=7)
plt.title('Red Wine Quality Distribution')
plt.xlabel('Quality Score')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
white['quality'].hist(color='grey', bins=7)
plt.title('White Wine Quality Distribution')
plt.xlabel('Quality Score')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# --- Citation ---
"""
Données fournies par:
- P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis (2009)
- Publication: Decision Support Systems
- DOI: 10.24432/C56S3T
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
"""

# Pour approfondir, consulte l’article original : http://www3.dsi.uminho.pt/pcortez/wine/

