import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder

# 1. Création d'un dataset d'exemple (Données mixtes)
data = {
    'Age': [25, 45, 35, 50, 23],
    'Salaire': [40000, 80000, 60000, 120000, 38000],
    'Ville': ['Paris', 'Lyon', 'Paris', 'Marseille', 'Lyon'],
    'Achat': ['Non', 'Oui', 'Oui', 'Oui', 'Non']
}
df = pd.DataFrame(data)
print("Dataset Original :")
print(df)

# --- PARTIE A : ENCODAGE DES VARIABLES CATÉGORIELLES ---

# 1. Label Encoding (pour la cible 'Achat' : Oui/Non -> 1/0)
le = LabelEncoder()
df['Achat'] = le.fit_transform(df['Achat'])

# 2. One-Hot Encoding (pour 'Ville' : crée une colonne par ville)
df_encoded = pd.get_dummies(df, columns=['Ville'], prefix='Ville')

print("\nAprès Encodage (Label + One-Hot) :")
print(df_encoded)

# --- PARTIE B : NORMALISATION / MISE À L'ÉCHELLE ---

# 1. Standardisation (StandardScaler) : Moyenne=0, Écart-type=1
scaler_std = StandardScaler()
df_std = df_encoded.copy()
df_std[['Age', 'Salaire']] = scaler_std.fit_transform(df_encoded[['Age', 'Salaire']])

print("\nAprès Standardisation (Z-score) :")
print(df_std.head(2))

# 2. Normalisation Min-Max (MinMaxScaler) : Valeurs entre [0, 1]
scaler_minmax = MinMaxScaler()
df_minmax = df_encoded.copy()
df_minmax[['Age', 'Salaire']] = scaler_minmax.fit_transform(df_encoded[['Age', 'Salaire']])

print("\nAprès Normalisation Min-Max :")
print(df_minmax.head(2))