# 📈 Travaux Pratiques : Machine Learning Prédictif & Analyse de Données

Ce dépôt contient une série de travaux pratiques (TPs) dédiés à l'apprentissage automatique supervisé et non supervisé. L'objectif est de maîtriser le cycle de vie complet d'un projet Data Science : de l'exploration de données (EDA) à l'optimisation avancée de modèles de boosting, en passant par le traitement des séries temporelles.

---

## 📑 Sommaire des TPs

### 1. Fondamentaux & Exploration
* **[TP Pandas — Dataset Iris](TP_Pandas_Iris.ipynb)** : Manipulation de DataFrames, statistiques descriptives et premières visualisations avec Matplotlib.
* **[TP2 — Clustering K-Means](TP2_KMeans.md)** : Segmentation de clientèle (Mall Customers), méthode du coude (Elbow) et interprétation des centroïdes.

### 2. Classification & Régression
* **[TP Classification](TP_Classification.md)** : Mise en œuvre d'Arbres de Décision, Random Forest et XGBoost sur le dataset Breast Cancer. Comparaison des métriques (Précision, Rappel, F1).
* **[TP Régression & Complexité](TP_Regression_CV.md)** : Prédiction de prix immobiliers (California Housing). Étude du compromis Biais/Variance et validation croisée (K-Fold).

### 3. Maîtrise de XGBoost
* **[TP4 — Validation & Early Stopping](TP4_XGBoost_Validation.md)** : Gestion du surapprentissage, réglage du `learning_rate` et utilisation d'un set de validation.
* **[TP7 — Objectifs & Métriques](TP7_XGBoost_Metrics.md)** : Fonctions de perte spécifiques (Poisson, Tweedie) et définition d'objectifs/métriques personnalisés (Gradients & Hessiens).

### 4. Séries Temporelles (Time Series)
* **[TP6.1 & 6.2 — Exploration et Stationnarité](TP6_TS_Exploration.md)** : Analyse de tendance, saisonnalité et tests de Dickey-Fuller (ADF).
* **[TP6.3 & 6.4 — ARIMA vs Approches Supervisées](TP6_TS_Models.md)** : Modélisation statistique (ARIMA) versus Machine Learning (Lags, fenêtres glissantes) sur la demande de vélos en libre-service.

### 5. Préparation de Données Avancée (Feature Engineering)
* **[TP8.1 — Sélection de Variables](TP8_1_Feature_Selection.md)** : Filtres univariés, Lasso (L1) et sélection séquentielle.
* **[TP8.2 — Encodage & Normalisation](TP8_2_Encoding_Scaling.md)** : One-Hot vs Target Encoding, et impact des Scalers (Standard, Robust, MinMax).
* **[TP8.3 — Valeurs Aberrantes (Outliers)](TP8_3_Outliers.md)** : Détection via IQR, Z-Score et Isolation Forest.
* **[TP8.4 — Classes Déséquilibrées](TP8_4_Imbalanced.md)** : Techniques de rééchantillonnage (SMOTE), pondération (`scale_pos_weight`) et courbes Precision-Recall.

---

## 🛠️ Installation et Environnement

### 1. Prérequis
* **Python 3.10+**
* **Bibliothèques clés** : `scikit-learn`, `pandas`, `numpy`, `xgboost`, `matplotlib`, `seaborn`, `statsmodels`, `imbalanced-learn`.

### 2. Installation rapide
```bash
# Création de l'environnement
python -m venv venv_predictif
source venv_predictif/bin/activate  # Windows: venv_predictif\Scripts\activate

# Installation des dépendances
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels imbalanced-learn category_encoders
