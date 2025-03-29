# Documentación de la API

## Clase ProductionModel

### Constructor

```python
ProductionModel(
    model_path: str,
    scaler_path: Optional[str] = None,
    threshold_rmse: float = 1000,
    threshold_mae: float = 500,
    drift_threshold: float = 0.1
)
```

**Parámetros:**
- `model_path`: Ruta al archivo del modelo entrenado
- `scaler_path`: Ruta al archivo del scaler (opcional)
- `threshold_rmse`: Umbral de alerta para RMSE
- `threshold_mae`: Umbral de alerta para MAE
- `drift_threshold`: Umbral para detección de drift

### Métodos

#### validate_data
```python
def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]
```

Valida los datos de entrada.

**Parámetros:**
- `df`: DataFrame con los datos a validar

**Retorna:**
- Tuple con (bool indicando si los datos son válidos, lista de errores)

**Validaciones:**
- Columnas requeridas
- Valores nulos
- Valores negativos
- Outliers

#### prepare_features
```python
def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame
```

Prepara las features para el modelo.

**Parámetros:**
- `df`: DataFrame con los datos

**Retorna:**
- DataFrame con features preparadas

**Features generadas:**
- Importe_FX_Volumen_Promedio
- Importe_FX_Spread
- Volumen_Ponderado_Spread
- Volumen_Importe_Ratio
- Spread_Volumen_Ratio

#### generate_predictions
```python
def generate_predictions(self, df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame
```

Genera predicciones para los próximos días.

**Parámetros:**
- `df`: DataFrame con datos históricos
- `horizon`: Número de días a predecir

**Retorna:**
- DataFrame con predicciones y intervalos de confianza

#### detect_data_drift
```python
def detect_data_drift(self, df: pd.DataFrame) -> Dict
```

Detecta drift en los datos de entrada.

**Parámetros:**
- `df`: DataFrame con los datos actuales

**Retorna:**
- Dict con métricas de drift

#### evaluate_predictions
```python
def evaluate_predictions(self, predictions: pd.DataFrame, actual_values: pd.Series) -> Dict
```

Evalúa las predicciones contra valores reales.

**Parámetros:**
- `predictions`: DataFrame con predicciones
- `actual_values`: Serie con valores reales

**Retorna:**
- Dict con métricas de evaluación (RMSE, MAE, R²)

#### save_predictions
```python
def save_predictions(self, predictions: pd.DataFrame) -> None
```

Guarda las predicciones en el historial.

**Parámetros:**
- `predictions`: DataFrame con predicciones

#### save_metrics
```python
def save_metrics(self, metrics: Dict) -> None
```

Guarda las métricas en el historial.

**Parámetros:**
- `metrics`: Dict con métricas de evaluación

#### get_feature_importance
```python
def get_feature_importance(self) -> pd.DataFrame
```

Obtiene la importancia de las features.

**Retorna:**
- DataFrame con importancia de features

#### generate_dashboard
```python
def generate_dashboard(self) -> None
```

Genera un dashboard interactivo con las métricas y predicciones.

## Estructura de Datos

### DataFrame de Entrada
```python
{
    'Importe FX': float,
    'Volumen_promedio_diario': float,
    'Spread_TC': float,
    'Volumen_Ponderado_5': float,
    'Volumen_diario': float,
    'Fecha de cierre': datetime
}
```

### DataFrame de Predicciones
```python
{
    'Fecha': datetime,
    'Predicción': float,
    'Intervalo_Inferior': float,
    'Intervalo_Superior': float,
    'Confianza': float
}
```

### Métricas de Evaluación
```python
{
    'rmse': float,
    'mae': float,
    'r2': float,
    'timestamp': str
}
```

## Ejemplos de Uso

### Inicialización
```python
model = ProductionModel(
    model_path='models/xgboost_model.joblib',
    scaler_path='models/scaler.joblib'
)
```

### Generación de Predicciones
```python
# Cargar datos
df = pd.read_csv('data/processed/features.csv')

# Generar predicciones
predictions = model.generate_predictions(df)

# Guardar resultados
model.save_predictions(predictions)
```

### Monitoreo
```python
# Detectar drift
drift_metrics = model.detect_data_drift(df)

# Generar dashboard
model.generate_dashboard()
```

## Manejo de Errores

El modelo maneja los siguientes tipos de errores:
- Validación de datos
- Errores de procesamiento
- Errores de guardado
- Errores de drift

Todos los errores son registrados en `logs/production.log`.

## Configuración

### Umbrales
- `threshold_rmse`: 1000
- `threshold_mae`: 500
- `drift_threshold`: 0.1

### Directorios
- Modelos: `models/`
- Datos: `data/`
- Resultados: `results/`
- Logs: `logs/` 