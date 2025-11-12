# Analyse approfondie de la base "Wine Quality"
# Source : UCI Machine Learning Repository
# Etude menée en 2009 par l'équipe du département Informatique, Universidade do Minho, Portugal
# Référence : Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# INFO SUR LA BASE
"""
Cette base regroupe des données physico-chimiques et des scores sensoriels liés à des vins portugais (blancs et rouges).
Objectif : Prédire la qualité perçue d'un vin (score 0-10) à partir de mesures chimiques.
Labo : Universidade do Minho, Portugal. Année : 2009.
Utilisation : Classification, régression, sélection de variables, data science, analyse prédictive.
Publication : Decision Support Systems
DOI : 10.24432/C56S3T
Licence : CC BY 4.0
"""

# 1. Chargement des données
red = pd.read_csv('winequality-red.csv', sep=';')
white = pd.read_csv('winequality-white.csv', sep=';')

# 2. Taille des jeux de données
print(f"Nombre d'échantillons vin

