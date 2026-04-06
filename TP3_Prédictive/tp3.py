import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Chargement des données [cite: 150]
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(f"Dimensions : {df.shape}")
print(f"Classes : {data.target_names}")
X = data.data
y = data.target

# Séparation 80% train / 20% test avec stratification [cite: 131]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Taille du train set : {len(X_train)}") 
print(f"Taille du test set : {len(X_test)}")  


# 1) Decision Tree [cite: 92]
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# 2) Random Forest [cite: 103]
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 3) XGBoost [cite: 115]
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

models = {"Decision Tree": dt_model, "Random Forest": rf_model, "XGBoost": xgb_model}
results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),   # [cite: 135]
        "Precision": precision_score(y_test, y_pred), # 
        "Recall": recall_score(y_test, y_pred),       # 
        "F1-score": f1_score(y_test, y_pred)          # [cite: 137]
    }

# Affichage des résultats
df_res = pd.DataFrame(results).T
print(df_res)

# Affichage de la matrice de confusion pour le meilleur modèle (ex: Random Forest)
y_pred_rf = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion - Random Forest")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.show()

# Importance des variables (Top 12) [cite: 156]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-12:]

plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [data.feature_names[i] for i in indices])
plt.title("Random Forest - Top 12 Importances")
plt.show()