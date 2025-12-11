<img src="https://s11.aconvert.com/convert/p3r68-cdx67/eoxpv-p0ojs.jpg"/>


# 📊 Projet Data Science : Reconnaissance de Chiffres Manuscrits (Digits)

## 🎯 Thématique
**Classification supervisée multi-classes** de chiffres manuscrits (0-9) à partir d'images 8×8 du dataset **Digits** de Scikit-Learn.

## 🛠️ Pipeline Complet Exécuté

```
1. SIMULATION → 5760 NaN injectés (5% par pixel)
2. NETTOYAGE → Imputation moyenne (SimpleImputer)
3. EDA → Stats + 3 visualisations (images/distributions/corrélations)
4. SPLIT → 80/20 stratifié (1438 Train / 359 Test)
5. MODÈLE → Random Forest (200 arbres)
6. AUDIT → Accuracy >95%, F1≈0.96, matrice confusion
```

## 📈 Résultats Clés
- **Accuracy globale** : **>95%**
- **F1-score macro** : **~0.96**
- **Forces** : Pipeline robuste, EDA riche, modèle performant
- **Limite** : Data leakage (imputation avant split ⚠️)

## 📁 Structure du Projet
```
├── 22007655_JAMAL_YASSINE.ipynb    ← Notebook Colab principal
├── Correction-Projet-Digits.md     ← Rapport détaillé (anatomie complète)
├── README.md                       ← Ce fichier
└── outputs/                        ← Visualisations générées
```



