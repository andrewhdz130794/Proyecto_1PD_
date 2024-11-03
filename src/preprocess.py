from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import yaml

# Cargar parámetros desde params.yaml
with open("C:/Users/andre/proyecto_1PD/params.yaml") as f:
    params = yaml.safe_load(f)

# Cargar datos desde la ruta especificada en params.yaml
data_path = params["data"]["path"]
data = pd.read_csv(data_path)

# Función de preprocesamiento
def preprocess_data(data):
    # Normalización para variables numéricas
    if params["preprocess"]["numeric"] == "normalize":
        numeric_features = data.select_dtypes(include=["int64", "float64"]).columns
        scaler = StandardScaler()
        data[numeric_features] = scaler.fit_transform(data[numeric_features])

    # Codificación OneHot para variables categóricas
    if params["preprocess"]["categorical"] == "onehot":
        categorical_features = data.select_dtypes(include=["object"]).columns
        encoder = OneHotEncoder(sparse_output=False)  # Cambiado a sparse_output
        encoded = encoder.fit_transform(data[categorical_features])
        data = data.drop(columns=categorical_features)
        data = pd.concat([data, pd.DataFrame(encoded)], axis=1)

    return data

# Aplicar preprocesamiento y guardar los datos procesados
processed_data = preprocess_data(data)
processed_data_path = "C:/Users/andre/proyecto_1PD/data/processed_data.csv"
processed_data.to_csv(processed_data_path, index=False)

print("Preprocesamiento completado. Datos guardados en:", processed_data_path)


