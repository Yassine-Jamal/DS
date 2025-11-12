# Analyse complète de la base de données "Wine Quality"
# Source : UCI Machine Learning Repository (2009)
# Étude réalisée par l'équipe informatique de l'Université do Minho, Portugal
# Référence : Cortez et al., 2009, Decision Support Systems

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Introduction ---
"""
Cette base de données présente des caractéristiques physico-chimiques et des notes de qualité de vins portugais, 
avec deux sous-ensembles distincts : vins rouges et vins blancs.
Le but principal est de prédire la qualité perçue des vins (évaluée de 0 à 10) à partir des propriétés mesurées en laboratoire.
Cette étude a été conçue pour servir à
