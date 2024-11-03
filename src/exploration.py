import pandas as pd
import yaml
import os

# Cargar parámetros desde params.yaml
with open("C:/Users/andre/proyecto_1PD/params.yaml") as f:
    params = yaml.safe_load(f)

# Usar la ruta completa para cargar los datos
data_path = params["data"]["path"]
data = pd.read_csv(data_path)

# Verificación de la carga de datos
print("Primeras filas del dataset:")
print(data.head())

# Guardar el archivo en la carpeta data usando la ruta completa
output_path = "C:/Users/andre/proyecto_1PD/data/data_limpia.csv"
data.to_csv(output_path, index=False)


