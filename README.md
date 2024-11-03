# Proyecto AutoML con DVC y Optuna

Este proyecto implementa un pipeline de Machine Learning usando DVC para la gestión de datos y el versionado de modelos, y Optuna/GridSearchCV para la optimización de hiperparámetros.

## Requisitos Previos

1. **Python 3.7 o superior**
2. **DVC** y **Git** instalados
3. Las dependencias de Python están listadas en `requirements.txt`

## Instalación y Configuración

1. **Clona el Repositorio**:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd proyecto_automl

# Instala las Dependencias:
pip install -r requirements.txt

# Inicializa DVC:
dvc init

# Ejecución del Pipeline de DVC
Para ejecutar el pipeline completo, usa el siguiente comando:
dvc repro

Este comando ejecutará todos los pasos definidos en el archivo dvc.yaml, desde la exploración de datos hasta la optimización de hiperparámetros.

# Configuración de Parámetros

Los parámetros del pipeline están definidos en params.yaml. Puedes modificar este archivo para ajustar los hiperparámetros y configuraciones del modelo.

# Comandos Específicos del Pipeline
dvc stage add -n explore -d data/raw_data.csv -o data/cleaned_data.csv python src/exploration.py

dvc stage add -n preprocess -d data/cleaned_data.csv -o data/processed_data.csv python src/preprocess.py

dvc stage add -n train -d data/processed_data.csv -o models/random_forest.pkl python src/train.py

dvc stage add -n evaluate -d data/test.csv -d models/random_forest.pkl -o results.csv python src/evaluate.py


# Resultados
Los resultados del pipeline se almacenan en los siguientes archivos:

optuna_results.csv: Contiene los mejores hiperparámetros y el MSE obtenido en la optimización de Optuna.
results.csv: Almacena las métricas de evaluación de los modelos, incluyendo el MSE y el R².

# Estructura del Proyecto

proyecto_1PD/
├── data/                       # Carpeta para los datasets 
├── models/                     # Carpeta para almacenar los modelos entrenados
├── outputs/                    # Carpeta para los resultados (opcional)
├── src/                        # Carpeta para los scripts del pipeline
│   ├── exploration.py          # Exploración de datos
│   ├── preprocess.py           # Preprocesamiento de datos
│   ├── train.py                # Entrenamiento de modelos
│   ├── evaluate.py             # Evaluación de modelos
│   └── optimize_optuna.py      # Optimización de hiperparámetros con Optuna
├── params.yaml                 # Archivo de parámetros para el pipeline
├── dvc.yaml                    # Archivo que define el pipeline de DVC
├── requirements.txt            # Lista de dependencias necesarias
└── README.md                   # Instrucciones para reproducir el proyecto
               
