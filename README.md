# Modelo de Predicción de Volúmenes FX

## Descripción
Sistema de predicción de volúmenes de operaciones FX utilizando machine learning, implementado en producción con monitoreo continuo y validaciones de seguridad.

## Características Principales
- Predicción de volúmenes FX usando XGBoost y Prophet
- Sistema de monitoreo en tiempo real
- Detección de drift de datos
- Validaciones de seguridad robustas
- Sistema de logging detallado
- Pruebas de integración completas

## Estructura del Proyecto

```
Fx/
├── src/
│   ├── models/
│   │   └── production_model.py
│   ├── data/
│   │   ├── data_processor.py
│   │   └── feature_engineering.py
│   └── utils/
│       └── security.py
├── tests/
│   ├── unit/
│   │   └── test_production_model.py
│   └── integration/
│       └── test_model_pipeline.py
├── docs/
│   ├── api.md
│   └── security.md
├── models/
│   ├── xgboost_model.joblib
│   └── scaler.joblib
├── data/
│   ├── raw/
│   │   └── Fluctua.xlsx    # Documento matriz con datos históricos
│   └── processed/
├── results/
│   ├── predictions_history.json
│   ├── metrics_history.json
│   └── model_performance.json
├── logs/
│   └── security/
├── security/
│   └── key.key
├── requirements.txt
├── setup.py
├── README.md
└── Proposal_Cinthia Fernandez.pdf  # Propuesta inicial del proyecto
```

## Datos y Documentación Base
- **Documento Matriz**: `data/raw/Fluctua.xlsx`
  - Pestaña: OPES
  - Encabezados: Segunda fila del archivo
  - Contiene datos históricos de operaciones FX

- **Propuesta Inicial**: `Proposal_Cinthia Fernandez.pdf`
  - Define los objetivos y alcance del proyecto
  - Especifica los requisitos y métricas de éxito

## Pipeline de Procesamiento

### 1. Preprocesamiento de Datos
- Limpieza de datos del archivo Fluctua.xlsx
- Manejo de valores faltantes y tipos de datos
- Validación de integridad de datos

### 2. Feature Engineering
- Creación de features temporales
- Normalización de datos
- Selección de features relevantes
- Validación de features

### 3. División de Datos
- Split temporal: 80% entrenamiento, 20% validación
- Validación cruzada temporal
- Preservación de la secuencia temporal

### 4. Modelos Implementados
- **XGBoost**
  - Optimización de hiperparámetros
  - Validación cruzada temporal
  - Métricas de rendimiento

- **Prophet**
  - Modelado de tendencias y estacionalidad
  - Predicciones a diferentes horizontes temporales
  - Comparación con XGBoost

### 5. Refinamiento y Entrenamiento
- Optimización de hiperparámetros
- Validación de rendimiento
- Selección del mejor modelo

## Métricas y Resultados
- RMSE: [valor]
- MAE: [valor]
- R²: [valor]
- MAPE: [valor]
- Comparación entre modelos:
  - XGBoost vs Prophet
  - Métricas por horizonte temporal

## Requisitos
- Python 3.8+
- Dependencias listadas en `requirements.txt`

## Instalación
1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/fx-prediction.git
cd fx-prediction
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso
1. Configurar las rutas de los modelos y datos en `config.py`
2. Ejecutar el modelo:
```bash
python src/models/production_model.py
```

## Monitoreo
- Logs de seguridad en `logs/security/`
- Métricas de rendimiento en `results/metrics_history.json`
- Predicciones históricas en `results/predictions_history.json`
- Análisis de rendimiento en `results/model_performance.json`

## Seguridad
- Encriptación de datos sensibles
- Validación de integridad de modelos
- Rotación automática de logs
- Permisos de archivos seguros

## Pruebas
```bash
# Ejecutar todas las pruebas
pytest

# Ejecutar pruebas unitarias
pytest tests/unit/

# Ejecutar pruebas de integración
pytest tests/integration/
```

## Contribución
Ver [CONTRIBUTING.md](CONTRIBUTING.md) para detalles sobre el proceso de contribución.

## Licencia
Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Changelog
Ver [CHANGELOG.md](CHANGELOG.md) para la lista de cambios.

## Preprocesamiento de Datos

El proyecto incluye un pipeline de preprocesamiento de datos que consta de dos componentes principales:

### 1. Limpieza de Datos (`src/preprocessing/data_cleaner.py`)
- Manejo de valores faltantes
- Corrección de tipos de datos
- Eliminación de duplicados
- Validación de datos

### 2. Ingeniería de Features (`src/preprocessing/feature_engineering.py`)
- Creación de features temporales
- Normalización de datos
- Selección de features
- Validación de features

### Ejecución del Preprocesamiento

```bash
# Limpieza de datos
python -m src.preprocessing.data_cleaner

# Ingeniería de features
python -m src.preprocessing.feature_engineering
```
