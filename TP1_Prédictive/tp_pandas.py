import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# 1. Charger le dataset Iris
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target_names[iris.target]

print("5 premières lignes :")
print(df.head())

# 2. Dimensions et statistiques
print("\nDimensions du dataset :")
print(df.shape)

print("\nDescription statistique :")
print(df.describe())

# 3. Renommer les colonnes
df.rename(columns={
    "sepal length (cm)": "SepalLengthCm",
    "sepal width (cm)": "SepalWidthCm",
    "petal length (cm)": "PetalLengthCm",
    "petal width (cm)": "PetalWidthCm"
}, inplace=True)

print("\nColonnes renommées :")
print(df.head())

# 4. Ajouter une colonne (rapport pétales)
df["PetalRatio"] = df["PetalLengthCm"] / df["PetalWidthCm"]

print("\nColonne PetalRatio ajoutée :")
print(df.head())

# 5. Supprimer une colonne inutile
df.drop(columns=["PetalRatio"], inplace=True)

print("\nColonne PetalRatio supprimée.")

# 6. Supprimer lignes où SepalLength < 5.0
df = df[df["SepalLengthCm"] >= 5.0]

print("\nDimensions après suppression des lignes :")
print(df.shape)

# 7. Filtrer uniquement l'espèce setosa
df_setosa = df[df["species"] == "setosa"]

print("\nLignes correspondant à l'espèce setosa :")
print(df_setosa.head())

# 8. Compter le nombre d'occurrences par espèce
print("\nNombre d'occurrences par espèce :")
print(df["species"].value_counts())

# 9. Visualisations
# 9.1 Histogramme
plt.hist(df["PetalLengthCm"], bins=20)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.title("Histogramme de la longueur des pétales")
plt.show()

# 9.2 Nuage de points
plt.scatter(df["SepalLengthCm"], df["SepalWidthCm"])
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Nuage de points : longueur vs largeur du sépale")
plt.show()

# 9.3 Boxplot par espèce
sns.boxplot(x="species", y="PetalLengthCm", data=df)
plt.title("Boxplot de la longueur des pétales par espèce")
plt.show()
