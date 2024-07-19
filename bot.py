# Importar la biblioteca OpenAI
from openai import OpenAI, OpenAIError

# Importar la biblioteca
import os
from os.path import join, dirname
from dotenv import load_dotenv
import pandas as pd

# API Key
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Crear el cliente de OpenAI
API_KEY = os.environ.get("API_KEY")
client = OpenAI(api_key=API_KEY)

# Importar las métricas

models_metrics = pd.read_csv("metrics/metrics.csv")

lineal_mse = models_metrics.loc[0, "MSE"]
lineal_mae = models_metrics.loc[0, "MAE"]
lineal_r2 = models_metrics.loc[0, "R2"]

tree_mse = models_metrics.loc[1, "MSE"]
tree_mae = models_metrics.loc[1, "MAE"]
tree_r2 = models_metrics.loc[1, "R2"]

kmeans_silhouette = models_metrics.loc[2, "Silhouette Score"]
pca_silhouette = models_metrics.loc[3, "Silhouette Score"]

# Importar los modelos

lineal_model = pd.read_pickle("models/linear_regression_model.pkl")
tree_model = pd.read_pickle("models/decision_tree_model.pkl")
kmeans_model = pd.read_pickle("models/kmeans_model.pkl")
pca_model = pd.read_pickle("models/pca_model.pkl")

# Función principal para el bot

def main():
    print("ChatGPT Consola. Escribe 'salir' para terminar la conversación.")
    while True:
        user_input = input("¿Qué modelo deseas consultar? ")
        if user_input.lower() == 'salir':
            print("¡Hasta luego!")
            break

        try:

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"El modelo de regresión lineal es este {lineal_model}."
                            f"El modelo de árbol de decisión es este {tree_model}."
                            f"El modelo de K-Means es este {kmeans_model}."
                            f"El modelo de PCA es este {pca_model}."
                            f"El modelo de regresión lineal tiene un MSE de {lineal_mse}, un MAE de {lineal_mae} y un R2 de {lineal_r2}."
                            f"El modelo de árbol de decisión tiene un MSE de {tree_mse}, un MAE de {tree_mae} y un R2 de {tree_r2}."
                            f"El modelo de K-Means tiene un Silhouette Score de {kmeans_silhouette * 100}%."
                            f"El modelo de PCA tiene un Silhouette Score de {pca_silhouette * 100}%."
                            f"Pregunta: {user_input}"
                        ),
                    }
                ],
                model="gpt-3.5-turbo",
            )

            # Extrae y muestra el contenido de la respuesta
            response_content = chat_completion.choices[0].message.content
            print(response_content)

        except OpenAIError as e:
            if e.code == 'rate_limit_exceeded':
                print("Has excedido la cuota de uso. Intenta más tarde.")
            else:
                print(f"Ocurrió un error: {e.message}")

if __name__ == "__main__":
    main()

