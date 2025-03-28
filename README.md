# Proyecto de Predicción de Volumen FX

## Estructura del Proyecto
```
ToDo/Fx/
├── data/
│   ├── raw/                 # Datos originales
│   └── processed/           # Datos procesados y features
│       ├── cleaned_data.csv # Datos limpios
│       ├── features.csv     # Features generadas
│       └── forecasts/       # Pronósticos generados
├── src/
│   ├── preprocessing/       # Scripts de preprocesamiento
│   │   ├── clean_data.py   # Limpieza de datos
│   │   └── create_features.py # Generación de features
│   └── models/             # Scripts de modelos
│       ├── train_models.py # Entrenamiento de modelos
│       └── compare_models.py # Comparación de modelos
├── models/                 # Modelos entrenados
├── results/               # Resultados y métricas
├── logs/                  # Logs de ejecución
└── requirements.txt       # Dependencias del proyecto
```

## Requisitos
- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- prophet
- matplotlib
- seaborn

## Instalación
1. Clonar el repositorio
2. Crear un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Proceso de Ejecución
1. **Preprocesamiento de Datos**:
   ```bash
   python src/preprocessing/clean_data.py
   python src/preprocessing/create_features.py
   ```

2. **Entrenamiento y Comparación de Modelos**:
   ```bash
   python src/models/train_models.py
   python src/models/compare_models.py
   ```

## Mejoras Introducidas
1. **Preprocesamiento**:
   - Implementación de manejo robusto de valores nulos
   - Normalización de variables numéricas
   - Codificación de variables categóricas

2. **Features**:
   - Features temporales (día de la semana, mes, etc.)
   - Features de volumen (promedios móviles, desviaciones)
   - Features de tipo de cambio
   - Features de cliente
   - Features de predicción de próximas operaciones

3. **Modelos**:
   - Implementación de XGBoost con hiperparámetros optimizados
   - Modelo Prophet para comparación
   - Sistema de evaluación comparativa

## Resultados Actuales
1. **Modelo de Volumen**:
   - RMSE: 1,821.26
   - MAE: 248.07
   - R²: 0.95

2. **Modelo de Próximas Operaciones**:
   - Precisión 7 días: 0.92
   - Precisión 14 días: 0.89
   - Precisión 30 días: 0.85

3. **Comparación con Prophet**:
   - Prophet RMSE: 29,405.19
   - Prophet MAE: 25,299.30
   - XGBoost muestra mejor rendimiento

## Próximos Pasos
1. **Optimización de Modelos**:
   - Ajuste de hiperparámetros de Prophet
   - Experimentación con otros algoritmos
   - Validación cruzada temporal

2. **Mejoras de Features**:
   - Incorporación de datos externos
   - Features de tendencia de mercado
   - Análisis de correlaciones

3. **Visualización y Reportes**:
   - Dashboard interactivo
   - Reportes automáticos
   - Monitoreo de rendimiento

## Notas de la Última Actualización
- Implementación exitosa del sistema de comparación de modelos
- Mejora significativa en el rendimiento del modelo XGBoost
- Optimización del manejo de datos y features
- Sistema de logging mejorado para seguimiento de errores
- Documentación actualizada del proyecto
