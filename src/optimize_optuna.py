import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Cargar datos de entrenamiento
data = pd.read_csv("C:/Users/andre/proyecto_1PD/data/processed_data.csv")
X = data.drop(columns=["price"])  
y = data["price"]

# Definir la función objetivo para Optuna
def objective(trial):
    max_depth = trial.suggest_int("max_depth", 5, 30)
    n_estimators = trial.suggest_int("n_estimators", 10, 200)

    # Crear el modelo con los hiperparámetros seleccionados
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    # Calcular el rendimiento del modelo con validación cruzada
    score = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=3)
    return score.mean()

# Crear y ejecutar el estudio de Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Obtener los mejores parámetros y resultados
best_params = study.best_params
best_score = study.best_value

# Crear un DataFrame para guardar los resultados
results = {
    "Model": ["Random Forest (Optimizado)"],
    "Best Max Depth": [best_params["max_depth"]],
    "Best N Estimators": [best_params["n_estimators"]],
    "Best Score (MSE)": [best_score]
}

results_df = pd.DataFrame(results)

# Guardar en un archivo CSV
results_df.to_csv("C:/users/andre/proyecto_1PD/outputs/optuna_results.csv", index=False)
print("Resultados de Optuna guardados en optuna_results.csv")
