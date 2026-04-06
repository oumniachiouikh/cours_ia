import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Chargement du dataset (California Housing)
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Séparation Entraînement / Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialisation des modèles
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 4. Validation Croisée (Cross-Validation)
# On évalue la performance sur 5 "folds" pour vérifier la robustesse
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')

print(f"R2 moyen (Cross-Val) - Random Forest: {np.mean(cv_scores):.4f}")

# 5. Entraînement et Prédiction
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# 6. Évaluation des performances
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nRésultats du modèle final :")
print(f"RMSE (Erreur quadratique moyenne) : {rmse:.4f}")
print(f"MAE (Erreur absolue moyenne) : {mae:.4f}")
print(f"R² Score (Coefficient de détermination) : {r2:.4f}")

# 7. Visualisation : Valeurs Réelles vs Prédictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.xlabel("Valeurs Réelles")
plt.ylabel("Prédictions")
plt.title("Régression : Réel vs Prédit")
plt.show()

# 8. Importance des variables (Feature Importance)
features = X.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 8))
plt.title('Importance des variables dans la régression')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importance Relative')
plt.show()