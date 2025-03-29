import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features temporales.
    
    Args:
        df: DataFrame con datos limpios
        
    Returns:
        DataFrame con features temporales agregadas
    """
    try:
        logger.info("Creando features temporales")
        
        # Extraer componentes de fecha
        df['Año'] = df['Fecha de cierre'].dt.year
        df['Mes'] = df['Fecha de cierre'].dt.month
        df['Día'] = df['Fecha de cierre'].dt.day
        df['Día_semana'] = df['Fecha de cierre'].dt.dayofweek
        df['Trimestre'] = df['Fecha de cierre'].dt.quarter
        
        # Crear features cíclicas
        df['Mes_sin'] = np.sin(2 * np.pi * df['Mes'] / 12)
        df['Mes_cos'] = np.cos(2 * np.pi * df['Mes'] / 12)
        df['Día_semana_sin'] = np.sin(2 * np.pi * df['Día_semana'] / 7)
        df['Día_semana_cos'] = np.cos(2 * np.pi * df['Día_semana'] / 7)
        
        # Crear features de días festivos
        df['Es_fin_semana'] = df['Día_semana'].isin([5, 6]).astype(int)
        df['Es_inicio_mes'] = (df['Día'] <= 5).astype(int)
        df['Es_fin_mes'] = (df['Día'] >= 25).astype(int)
        
        return df
        
    except Exception as e:
        logger.error(f"Error al crear features temporales: {str(e)}")
        raise

def create_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features relacionadas con el volumen.
    
    Args:
        df: DataFrame con datos limpios
        
    Returns:
        DataFrame con features de volumen agregadas
    """
    try:
        logger.info("Creando features de volumen")
        
        # Calcular ratios
        df['Volumen_Importe_Ratio'] = df['Volumen_diario'] / df['Importe FX']
        df['Volumen_Promedio_Ratio'] = df['Volumen_diario'] / df['Volumen_promedio_diario']
        df['Spread_Volumen_Ratio'] = df['Spread_TC'] / df['Volumen_diario']
        
        # Calcular promedios móviles
        for window in [3, 5, 7, 14]:
            df[f'Volumen_MA_{window}'] = df.groupby('Codigo del Cliente (IBS)')['Volumen_diario'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'Importe_MA_{window}'] = df.groupby('Codigo del Cliente (IBS)')['Importe FX'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # Calcular desviaciones
        for window in [3, 5, 7]:
            df[f'Volumen_Std_{window}'] = df.groupby('Codigo del Cliente (IBS)')['Volumen_diario'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        return df
        
    except Exception as e:
        logger.error(f"Error al crear features de volumen: {str(e)}")
        raise

def create_client_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features relacionadas con el cliente.
    
    Args:
        df: DataFrame con datos limpios
        
    Returns:
        DataFrame con features de cliente agregadas
    """
    try:
        logger.info("Creando features de cliente")
        
        # Calcular estadísticas por cliente
        client_stats = df.groupby('Codigo del Cliente (IBS)').agg({
            'Volumen_diario': ['mean', 'std', 'max', 'min'],
            'Importe FX': ['mean', 'std', 'max', 'min'],
            'Spread_TC': ['mean', 'std']
        }).reset_index()
        
        # Renombrar columnas
        client_stats.columns = ['Codigo del Cliente (IBS)'] + [
            f'Cliente_{col[0]}_{col[1]}' for col in client_stats.columns[1:]
        ]
        
        # Unir estadísticas al DataFrame principal
        df = df.merge(client_stats, on='Codigo del Cliente (IBS)', how='left')
        
        # Crear features categóricas
        df['Categoria_Cliente_Encoded'] = pd.Categorical(df['Categoria_Cliente']).codes
        
        return df
        
    except Exception as e:
        logger.error(f"Error al crear features de cliente: {str(e)}")
        raise

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea todas las features necesarias para el modelo.
    
    Args:
        df: DataFrame con datos limpios
        
    Returns:
        DataFrame con todas las features
    """
    try:
        logger.info("Iniciando creación de features")
        
        # Crear copia para no modificar los datos originales
        df_features = df.copy()
        
        # Crear features temporales
        df_features = create_temporal_features(df_features)
        
        # Crear features de volumen
        df_features = create_volume_features(df_features)
        
        # Crear features de cliente
        df_features = create_client_features(df_features)
        
        # Normalizar features numéricas
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df_features[numeric_columns] = scaler.fit_transform(df_features[numeric_columns])
        
        # Guardar scaler
        scaler_path = Path('models/scaler.joblib')
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler guardado en {scaler_path}")
        
        # Guardar features
        output_path = Path('data/processed/features.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(output_path, index=False)
        logger.info(f"Features guardadas en {output_path}")
        
        return df_features
        
    except Exception as e:
        logger.error(f"Error en creación de features: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Cargar datos limpios
        input_path = Path('data/processed/cleaned_data.csv')
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de datos limpios en {input_path}")
        
        df = pd.read_csv(input_path)
        df['Fecha de cierre'] = pd.to_datetime(df['Fecha de cierre'])
        logger.info(f"Datos limpios cargados desde {input_path}")
        
        # Crear features
        df_features = create_features(df)
        
        # Mostrar resumen
        logger.info("\nResumen de features:")
        logger.info(f"Total de features creadas: {len(df_features.columns)}")
        logger.info("\nTipos de features:")
        logger.info(df_features.dtypes.value_counts())
        
    except Exception as e:
        logger.error(f"Error en ejecución principal: {str(e)}")
        raise
