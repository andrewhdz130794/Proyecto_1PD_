from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import yaml
import pickle

# Cargar parámetros desde params.yaml
with open("C:/Users/andre/proyecto_1PD/params.yaml") as f:
    params = yaml.safe_load(f)

# Cargar datos procesados
data = pd.read_csv("C:/Users/andre/proyecto_1PD/data/processed_data.csv")

# Separar características (X) y variable objetivo (y)
# Cambia 'target' al nombre de tu columna objetivo real
X = data.drop(columns=["target"])  # Ajusta 'target' según tu dataset
y = data["target"]

# Definir los modelos
models = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(
        n_estimators=params["hyperparameters"]["random_forest"]["n_estimators"],
        max_depth=params["hyperparameters"]["random_forest"]["max_depth"]
    ),
    "gradient_boosting": GradientBoostingRegressor(
        learning_rate=params["hyperparameters"]["gradient_boosting"]["learning_rate"],
        n_estimators=params["hyperparameters"]["gradient_boosting"]["n_estimators"]
    )
}

# Entrenar y evaluar cada modelo
results = {}
for model_name, model in models.items():
    print(f"Entrenando {model_name}...")
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    results[model_name] = mse
    
    # Guardar el modelo entrenado
    model_path = f"C:/Users/andre/proyecto_1PD/models/{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"{model_name} guardado en {model_path} con MSE: {mse}")

# Mostrar los resultados
print("\nResultados de entrenamiento:")
for model_name, mse in results.items():
    print(f"{model_name}: MSE = {mse}")
