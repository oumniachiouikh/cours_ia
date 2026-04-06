import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Chargement du jeu de données
url = "https://raw.githubusercontent.com/satishgunjal/datasets/master/Mall_Customers.csv"
df = pd.read_csv(url)

# Vérification des dimensions et des premières lignes
print(f"Dimensions du tableau : {df.shape}")
print(df.head())


## 2. Exploration et Préparation des Données

# Statistiques descriptives et informations générales
print(df.describe())
print(df.info())

# Renommage des colonnes
df.rename(columns={
    'Annual Income (k$)': 'AnnualIncome', 
    'Spending Score (1-100)': 'SpendingScore'
}, inplace=True)

# Visualisation initiale (Nuage de points)
plt.figure(figsize=(8, 6))
plt.scatter(df['AnnualIncome'], df['SpendingScore'], color='gray', alpha=0.6)
plt.title('Répartition des clients (Revenu vs Score de dépenses)')
plt.xlabel('Revenu annuel (k$)')
plt.ylabel('Score de dépenses (1-100)')
plt.show()

# Construction de la matrice de caractéristiques X
X = df.loc[:, ["AnnualIncome", "SpendingScore"]].values
print(f"Premières lignes de X :\n{X[:5]}")

## 3. Détermination de k (Méthode du Coude)
wcss = [] # Liste pour stocker l'inertie (Within-Cluster Sum of Square)

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='random', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Tracé de la courbe
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Méthode du coude pour choisir k')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Inertie intra-cluster')
plt.grid(True)
plt.show()

## 4. Entraînement et Visualisation des Clusters

# Entraînement avec k=5
kmeans_final = KMeans(n_clusters=5, init='random', random_state=42)
y_kmeans = kmeans_final.fit_predict(X)

# Ajout des étiquettes au DataFrame
df['Cluster'] = y_kmeans

# Analyse des caractéristiques par segment
analysis = df.groupby('Cluster').agg({
    'AnnualIncome': ['mean', 'count'],
    'SpendingScore': 'mean'
})
print("Analyse par cluster :\n", analysis)

# Visualisation finale
plt.figure(figsize=(10, 7))
colors = ['orange', 'blue', 'green', 'red', 'purple']
for i in range(5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')

plt.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1], 
            s=200, c='black', marker='X', label='Centroïdes')
plt.title('Segmentation finale des clients')
plt.xlabel('Revenu annuel (k$)')
plt.ylabel('Score de dépenses')
plt.legend()
plt.show()