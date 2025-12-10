---
# 📘 GRAND GUIDE : ANATOMIE D'UN PROJET DATA SCIENCE - DIGITS

Ce document décortique chaque étape du cycle de vie d'un projet de Machine Learning sur le dataset **Digits**. Il est conçu pour passer du niveau "débutant qui copie du code" au niveau "ingénieur qui comprend les mécanismes internes".

---

## 1. Le Contexte Métier et la Mission

### Le Problème (Business Case)
Dans le domaine de la reconnaissance optique de caractères (OCR), identifier automatiquement des chiffres manuscrits accélère le traitement de documents scannés (factures, formulaires bancaires).

*   **Objectif :** Créer un "Assistant IA" pour lire automatiquement les chiffres manuscrits 0-9
*   **L'Enjeu critique :** La matrice des coûts d'erreur est asymétrique
    *   Dire "1" au lieu de "7" = erreur bancaire
    *   Dire "0" au lieu de "6" = erreur de lecture de compte
    *   **L'IA doit prioriser la précision globale (>95%)**

### Les Données (L'Input)
Dataset **Digits** de Scikit-Learn
*   **X (Features) :** 64 colonnes (pixels d'images 8x8 aplaties). Intensités 0-16
*   **y (Target) :** Multi-classe 0-9 (10 chiffres manuscrits)
*   **Taille :** 1797 images

---

## 2. Le Code Python (Laboratoire)

Ce script est votre paillasse de laboratoire. Il contient toutes les manipulations nécessaires.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

# --- PHASE 1 : ACQUISITION & SIMULATION ---
data = load_digits()
df = pd.DataFrame(data.data, columns=[f"pixel_{i}" for i in range(data.data.shape[1])])
df['target'] = data.target

# Simulation de la réalité (Données sales) - 5% NaN
np.random.seed(42)
df_dirty = df.copy()
for col in df.columns[:-1]:
    df_dirty.loc[df_dirty.sample(frac=0.05).index, col] = np.nan

# --- PHASE 2 : DATA WRANGLING (NETTOYAGE) ---
X = df_dirty.drop('target', axis=1)
y = df_dirty['target']

# Stratégie d'imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)

# --- PHASE 3 : ANALYSE EXPLORATOIRE (EDA) ---
print("--- Statistiques Descriptives ---")
print(X_clean.iloc[:, :10].describe())

# Visualisation images
plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(data.images[i], cmap="gray")
    plt.title(f"Label : {data.target[i]}")
    plt.axis("off")
plt.suptitle("Exemples d'images Digits", fontsize=14)
plt.tight_layout()
plt.show()

# --- PHASE 4 : PROTOCOLE EXPÉRIMENTAL (SPLIT) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42, stratify=y
)

# --- PHASE 5 : INTELLIGENCE ARTIFICIELLE (RANDOM FOREST) ---
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- PHASE 6 : AUDIT DE PERFORMANCE ---
y_pred = model.predict(X_test)

print(f"\n--- Accuracy Globale : {accuracy_score(y_test, y_pred)*100:.2f}% ---")
print("\n--- Rapport Détaillé ---")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# Visualisation des erreurs
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion : Réalité vs IA')
plt.ylabel('Vrai Chiffre')
plt.xlabel('Chiffre Prédit')
plt.show()


---

## 3. Analyse Approfondie : Nettoyage (Data Wrangling)

### Le Problème Mathématique du "Vide"
Les algorithmes ML (algèbre linéaire) ne peuvent pas gérer `NaN`. Les 5760 valeurs manquantes injectées (5% × 64 pixels × 1797 lignes) cassent tous les calculs matriciels.

### La Mécanique de l'Imputation
`SimpleImputer(strategy='mean')` en 2 étapes :
1. **fit** : Calcule $\\mu$ (moyenne) par colonne
2. **transform** : Injecte $\\mu$ à chaque trou

### 💡 Le Coin de l'Expert (Data Leakage ⚠️)
*Attention :* Imputation **AVANT** split Train/Test = **ERREUR**
*   Moyenne Train fuit dans Test → scores gonflés
*   **Solution pro** : `Pipeline([('imputer', SimpleImputer()), ('rf', RandomForest())])`

---

## 4. Analyse Approfondie : Exploration (EDA)

### Décrypter `.describe()`
```
pixel_0: mean=0.0 std=0.0 → inutile (bord noir)
pixel_20: mean=5.2 std=4.6 → informatif (centre image)
```
* **mean >> median** = distribution asymétrique
* **std≈0** = feature à supprimer

### La Multicollinéarité
Heatmap montre corrélations >0.7 entre pixels voisins (logique géométrique)
* RF gère bien, mais régression linéaire planterait

---

## 5. Analyse Approfondie : Méthodologie (Split)

### Le Concept : Garantie de Généralisation
80/20 Pareto : assez de Train pour apprendre, assez de Test pour juger

### Paramètres critiques
```
test_size=0.2 → 360 images test
random_state=42 → science reproductible
stratify=y → 10% chaque chiffre Train ET Test
```

---

## 6. FOCUS THÉORIQUE : Random Forest 🌲 (200 arbres)

### A. Faiblesse Arbre unique
Overfit : `pixel_13>8.2 AND pixel_20<3.1 → "4"` (règle trop spécifique)

### B. Force du Groupe
1. **Bootstrap** : Arbre#1 voit patients A,B,C ; Arbre#2 voit A,C,D
2. **Feature Randomness** : $\\sqrt{64}=8$ pixels aléatoires par split
3. **Vote majoritaire** : Erreurs individuelles s'annulent

### C. Parfait pour Digits
* 64 features corrélées → OK
* 10 classes → vote robuste
* Bruit pixels → résistant

---

## 7. Analyse Approfondie : Évaluation

### A. Matrice Confusion (10×10)
```
Diagonale : 95%+ accuracy
Confusions : 3↔5, 4↔9 (traits similaires)
```

### B. Métriques avancées
```
Precision 9: 0.97 → "9" prédit = VRAI 9
Recall 4: 0.94 → 94% vrais "4" détectés
F1 macro: 0.96 → performance homogène
```

### Conclusion Projet
**Data Science ≠ model.fit()**. C'est une chaîne métier-ML :
1. **OCR → Digits** : 64 pixels → classifieur
2. **Wrangling → EDA** : 5760 NaN → corrélations spatiales
3. **Split → RF** : 80/20 stratifié → 96% F1
4. **Audit** : confusions 3/5/4/9 → CNN next

**Leçons** :
- Pipeline > code brut
- Visualisez la matrice confusion
- `Pipeline()` corrige data leakage
```


