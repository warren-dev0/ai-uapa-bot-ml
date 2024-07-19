# Importar librerías necesarias
import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score

# import core_samples_mask
from sklearn import metrics

# Cargar el conjunto de datos
# Usar un conjunto de datos público o simulado. Aquí usaremos datos simulados.
# Preprocesar los datos

# Get the absolute path of the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute file path to the CSV file
file_path = os.path.join(current_directory, "datos.csv")

# Load the CSV file
data = pd.read_csv(file_path)

# No hay valores faltantes en este conjunto de datos simulado

# Normalización
data['Publicidad'] = (data['Publicidad'] - data['Publicidad'].mean()) / data['Publicidad'].std()
# Dividir el conjunto de datos
X = data[['Publicidad']]
y = data['Ventas']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar algoritmos de aprendizaje supervisado
# Regresión Lineal

# Crear el modelo
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# Evaluar el modelo
mse_lin = mean_squared_error(y_test, y_pred_lin)
mae_lin = mean_absolute_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)


# Árboles de Decisión
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
y_pred_tree = tree_reg.predict(X_test)

# Evaluar el modelo
mse_tree = mean_squared_error(y_test, y_pred_tree)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

#Aprendizaje no supervisado
#Primer algoritmo utilizado (Kmean)

# Crear el modelo
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)
y_kmeans = kmeans.predict(X_test)

# Evaluar el modelo
silhouette = silhouette_score(X_test, y_kmeans)

# Segundo algoritmo utilizado (PCA)

# Crear el modelo
datairis = load_iris()
X_pca = datairis.data
y = datairis.target
pca = PCA(n_components=2)
X_r = pca.fit_transform(X_pca)

# Evaluar el modelo
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio of each principal component: {explained_variance}")
print(f"Total explained variance: {explained_variance.sum()}")

# Guardar los modelos
# Guardar los modelos en un archivo binario usando la biblioteca pickle
# Get the absolute path of the models directory
models_directory = os.path.join(current_directory, "models")

# Create the models directory if it does not exist
if not os.path.exists(models_directory):
    os.makedirs(models_directory)

# Construct the absolute file path to the linear regression model file
lin_reg_model_path = os.path.join(models_directory, "linear_regression_model.pkl")

# Save the linear regression model
with open(lin_reg_model_path, "wb") as file:
    pickle.dump(lin_reg, file)

# Construct the absolute file path to the decision tree model file
tree_reg_model_path = os.path.join(models_directory, "decision_tree_model.pkl")

# Save the decision tree model
with open(tree_reg_model_path, "wb") as file:
    pickle.dump(tree_reg, file)

# Construct the absolute file path to the kmeans model file
kmeans_model_path = os.path.join(models_directory, "kmeans_model.pkl")

# Save the kmeans model
with open(kmeans_model_path, "wb") as file:
    pickle.dump(kmeans, file)

# Construct the absolute file path to the pca model file
pca_model_path = os.path.join(models_directory, "pca_model.pkl")

# Save the pca model
with open(pca_model_path, "wb") as file:
    pickle.dump(pca, file)

# Visualización de resultados
# Visualizar los resultados de la regresión lineal
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred_lin, color='blue', linewidth=3)
plt.title('Regresión Lineal')
plt.xlabel('Publicidad')
plt.ylabel('Ventas')
plt.show()

# Visualizar los resultados del árbol de decisión
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred_tree, color='blue', linewidth=3)
plt.title('Árbol de Decisión')
plt.xlabel('Publicidad')
plt.ylabel('Ventas')
plt.show()

# Visualizar los resultados de KMeans
plt.scatter(X_test, y_test, c=y_kmeans, cmap='viridis')
plt.title('KMeans')
plt.xlabel('Publicidad')
plt.ylabel('Ventas')
plt.show()

# Visualizar los resultados de PCA
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], datairis.target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Imprimir las métricas de evaluación
print("Métricas de Regresión Lineal")
print("MSE:", mse_lin)
print("MAE:", mae_lin)
print("R2:", r2_lin)
print()

print("Métricas de Árbol de Decisión")
print("MSE:", mse_tree)
print("MAE:", mae_tree)
print("R2:", r2_tree)
print()

print("Métrica de KMeans")
print("Silhouette Score:", silhouette)
print()

print("Métrica de PCA")
print("Silhouette Score:", silhouette)
print()


# Guardar las métricas de evaluación
# Guardar las métricas de evaluación en un archivo CSV
# Get the absolute path of the metrics directory

metrics_directory = os.path.join(current_directory, "metrics")

# Create the metrics directory if it does not exist

if not os.path.exists(metrics_directory):
    os.makedirs(metrics_directory)

# Construct the absolute file path to the metrics file
metrics_file_path = os.path.join(metrics_directory, "metrics.csv")

# Create a DataFrame with the metrics
metrics_data = {
    "Model": ["Regresión Lineal", "Árbol de Decisión", "KMeans", "PCA"],
    "MSE": [mse_lin, mse_tree, 0, 0],
    "MAE": [mae_lin, mae_tree, 0, 0],
    "R2": [r2_lin, r2_tree, 0, 0],
    "Silhouette Score": [0, 0, silhouette, silhouette]
}

metrics_df = pd.DataFrame(metrics_data)

# Save the metrics to a CSV file
metrics_df.to_csv(metrics_file_path, index=False)

# Visualización de las métricas
# Visualizar las métricas de evaluación en un gráfico de barras
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
barplot = sns.barplot(data=metrics_df, x="Model", y="MSE")
plt.title("MSE de los Modelos")
plt.show()

plt.figure(figsize=(10, 6))
barplot = sns.barplot(data=metrics_df, x="Model", y="MAE")
plt.title("MAE de los Modelos")
plt.show()

plt.figure(figsize=(10, 6))
barplot = sns.barplot(data=metrics_df, x="Model", y="R2")
plt.title("R2 de los Modelos")
plt.show()

plt.figure(figsize=(10, 6))
barplot = sns.barplot(data=metrics_df, x="Model", y="Silhouette Score")
plt.title("Silhouette Score de los Modelos")
plt.show()


# Fin del script