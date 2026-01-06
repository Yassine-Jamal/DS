# Dataset utilisé : Digits de Scikit-Learn

- Source : module `sklearn.datasets`, fonction `load_digits()`
- Référence : https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset
- Format :
  - 1 797 images de chiffres manuscrits
  - 8 x 8 pixels par image (64 features d’intensité)
  - Intensités codées de 0 à 16
  - Cible `target` : 10 classes (chiffres 0 à 9)
- Chargement :
  ```python
  from sklearn.datasets import load_digits
  digits = load_digits()
  X = digits.data      # shape (1797, 64)
  y = digits.target    # labels 0–9

  Particularité du projet :

5 % de valeurs manquantes artificielles injectées par feature pour simuler des données “sales”
Imputation par la moyenne via SimpleImputer de Scikit-Learn


Dans ton notebook / script, ajoute la sauvegarde des figures clés, par exemple :[1][3]

```python
import os
os.makedirs("figures", exist_ok=True)

# Exemples d’images
plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title(f"Label: {digits.target[i]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("figures/examples_digits.png", dpi=300, bbox_inches="tight")

# Matrice de confusion
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel("Prédiction")
plt.ylabel("Réalité")
plt.title("Matrice de confusion - Digits Random Forest")
plt.tight_layout()
plt.savefig("figures/confusion_matrix_digits.png", dpi=300, bbox_inches="tight")
