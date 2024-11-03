import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score

# Cargar el modelo entrenado
model_path = "C:/Users/andre/proyecto_1PD/models/best_random_forest.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Cargar los datos de prueba
test_data = pd.read_csv(("C:/Users/andre/proyecto_1PD/data/processed_data.csv"))

# Definir X_test e y_test
# Cambia 'price' por el nombre de la columna objetivo en tu dataset
X_test = test_data.drop(columns=["price"])
y_test = test_data["price"]

# Realizar predicciones
predictions = model.predict(X_test)

# Calcular métricas
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Mostrar las métricas
print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")


# Definir las métricas en un diccionario
results = {
    "Model": ["Random Forest (Optimizado)"],
    "Mean Squared Error (MSE)": [mse],
    "R^2 Score": [r2]
}

# Crear un DataFrame con los resultados
results_df = pd.DataFrame(results)

# Guardar en un archivo CSV (modo "append" para agregar si ya existe)
results_path = f"C:/users/andre/proyecto_1PD/outputs/results.csv"
results_df.to_csv(results_path, mode='a', index=False, header=not pd.io.common.file_exists(results_path))

print(f"Resultados guardados en {results_path}")
