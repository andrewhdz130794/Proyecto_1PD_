from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import GridSearchCV
data = pd.read_csv("C:/Users/andre/proyecto_1PD/data/processed_data.csv")
X = data.drop(columns=["price"])  
y = data["price"]
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)
print("Mejores hiperparámetros para Random Forest:", grid_search.best_params_)

