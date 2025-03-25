import pandas as pd
import numpy as np

# Función para crear características rezagadas (Lag Features)
def create_lag_features(data, lag_columns, lags=[1, 7, 14]):
    """Crea características rezagadas para capturar patrones temporales."""
    for col in lag_columns:
        for lag in lags:
            data[f'{col}_lag_{lag}'] = data[col].shift(lag)
    return data

# Función para crear características de ventana móvil (Rolling Features)
def create_rolling_features(data, rolling_columns, windows=[7, 14, 30]):
    """Crea promedios móviles para suavizar fluctuaciones y detectar tendencias."""
    for col in rolling_columns:
        for window in windows:
            data[f'{col}_rolling_{window}'] = data[col].rolling(window=window).mean()
    return data

# Función para agregar características basadas en la fecha
def create_date_features(data, date_column='Fecha'):
    """Extrae información útil de una columna de fecha."""
    if date_column in data.columns:
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        data['year'] = data[date_column].dt.year
        data['month'] = data[date_column].dt.month
        data['day'] = data[date_column].dt.day
        data['dayofweek'] = data[date_column].dt.dayofweek
        data['is_weekend'] = data['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    return data

# Función para codificar variables categóricas
def encode_categorical_features(data, categorical_columns):
    """Codifica columnas categóricas usando One-Hot Encoding."""
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    return data

# Función principal para aplicar toda la ingeniería de características
def feature_engineering(data):
    """Aplica todas las técnicas de ingeniería de características."""
    
    # Identificar columnas numéricas y categóricas
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Aplicar las transformaciones
    data = create_lag_features(data, numeric_cols)
    data = create_rolling_features(data, numeric_cols)
    data = create_date_features(data)
    data = encode_categorical_features(data, categorical_cols)

    # Eliminar filas con valores nulos generados por las características rezagadas
    data.dropna(inplace=True)

    return data

# Bloque principal
if __name__ == "__main__":
    # Cargar el dataset limpio
    file_path = './data/cleaned_data.csv'
    data = pd.read_csv(file_path)

    # Aplicar ingeniería de características
    enriched_data = feature_engineering(data)

    # Guardar el dataset enriquecido
    enriched_data.to_csv('./data/enriched_data.csv', index=False)

    print("\n✅ Ingeniería de características completada. El archivo enriquecido se ha guardado como 'enriched_data.csv'")
