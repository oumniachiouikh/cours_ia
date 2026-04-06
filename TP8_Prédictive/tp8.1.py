import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# 1. Chargement des données
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
print(f"Nombre initial de variables : {X.shape[1]}")

# 2. Analyse de Corrélation
corr_matrix = X.corr().abs()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title("Matrice de Corrélation des variables")
plt.show()

# 3. Suppression des variables fortement corrélées (> 0.90)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
print(f"Variables supprimées (corrélation > 0.90) : {to_drop}")
X_reduced = X.drop(columns=to_drop)
print(f"Nombre de variables après filtrage par corrélation : {X_reduced.shape[1]}")

# 4. Sélection basée sur l'importance (Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_reduced, y)

selector = SelectFromModel(rf, prefit=True)
X_important = selector.transform(X_reduced)

# Récupération des noms des colonnes conservées
selected_features = X_reduced.columns[selector.get_support()]

print(f"\nVariables finales conservées par importance :")
print(list(selected_features))
print(f"Nombre final de variables : {len(selected_features)}")

# 5. Visualisation de l'importance des variables finales
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Importance des variables (Top 10)")
plt.bar(range(10), importances[indices[:10]])
plt.xticks(range(10), X_reduced.columns[indices[:10]], rotation=45)
plt.show()