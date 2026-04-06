import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

# 1. Préparation des données
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 2. Définition d'une fonction d'objectif personnalisée (Weighted Logloss)
def custom_weighted_logloss(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds)) # Transformation Sigmoid
    weight = 5.0    
    grad = preds * (weight * (1 - labels) + labels) - (weight * (1 - labels))
    hess = preds * (1 - preds) * (weight * (1 - labels) + labels)  
    return grad, hess

# 3. Définition d'une métrique d'évaluation personnalisée
def custom_recall_metric(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    preds_binary = (preds > 0.5).astype(int)
    recall = recall_score(labels, preds_binary)
    return 'custom_recall', recall

# 4. Entraînement du modèle avec les fonctions personnalisées
import xgboost as xgb

params = {
    'max_depth': 4,
    'eta': 0.1,
    'verbosity': 0
}
print("Entraînement en cours...")
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=50,
    obj=custom_weighted_logloss,
    custom_metric=custom_recall_metric,
    maximize=True
)
print("Entraînement terminé.")

# 5. Prédictions et Évaluation
raw_preds = bst.predict(dtest)
final_preds_proba = 1.0 / (1.0 + np.exp(-raw_preds))
final_preds = (final_preds_proba > 0.5).astype(int)

# 6. Comparaison des résultats
print("\n--- Résultats Finaux ---")
print(f"Accuracy: {accuracy_score(y_test, final_preds):.4f}")
print(f"Recall (Sensibilité): {recall_score(y_test, final_preds):.4f}")
print("\nMatrice de Confusion :")
print(confusion_matrix(y_test, final_preds))