import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

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
    Crea características temporales a partir de las fechas.
    
    Args:
        df: DataFrame con datos limpios
        
    Returns:
        DataFrame con características temporales adicionales
    """
    try:
        logger.info("Creando características temporales")
        
        # Asegurar que la columna de fecha es datetime
        df['Fecha de cierre'] = pd.to_datetime(df['Fecha de cierre'])
        
        # Características temporales básicas
        df['Año'] = df['Fecha de cierre'].dt.year
        df['Mes'] = df['Fecha de cierre'].dt.month
        df['Día'] = df['Fecha de cierre'].dt.day
        df['Día_semana'] = df['Fecha de cierre'].dt.dayofweek
        df['Trimestre'] = df['Fecha de cierre'].dt.quarter
        
        # Características cíclicas
        df['Mes_sin'] = np.sin(2 * np.pi * df['Mes'] / 12)
        df['Mes_cos'] = np.cos(2 * np.pi * df['Mes'] / 12)
        df['Día_semana_sin'] = np.sin(2 * np.pi * df['Día_semana'] / 7)
        df['Día_semana_cos'] = np.cos(2 * np.pi * df['Día_semana'] / 7)
        
        # Indicadores de fin de mes/semana
        df['Es_fin_mes'] = df['Fecha de cierre'].dt.is_month_end.astype(int)
        df['Es_fin_semana'] = (df['Día_semana'] >= 5).astype(int)
        
        return df
        
    except Exception as e:
        logger.error(f"Error al crear características temporales: {str(e)}")
        raise

def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea características estadísticas por cliente.
    
    Args:
        df: DataFrame con datos limpios
        
    Returns:
        DataFrame con características estadísticas adicionales
    """
    try:
        logger.info("Creando características estadísticas")
        
        # Agrupar por cliente
        client_groups = df.groupby('Codigo del Cliente (IBS)')
        
        # Estadísticas de volumen
        df['Volumen_std'] = client_groups['Importe FX'].transform('std')
        df['Volumen_max'] = client_groups['Importe FX'].transform('max')
        df['Volumen_min'] = client_groups['Importe FX'].transform('min')
        df['Volumen_mediana'] = client_groups['Importe FX'].transform('median')
        
        # Estadísticas de spread
        df['Spread_std'] = client_groups['Spread_TC'].transform('std')
        df['Spread_max'] = client_groups['Spread_TC'].transform('max')
        df['Spread_min'] = client_groups['Spread_TC'].transform('min')
        df['Spread_mediana'] = client_groups['Spread_TC'].transform('median')
        
        # Frecuencia de operaciones
        df['Operaciones_por_mes'] = client_groups['Importe FX'].transform('count')
        
        # Tendencia de volumen (últimos 30 días vs anteriores)
        df['Fecha_ordinal'] = df['Fecha de cierre'].astype(np.int64) // 10**9
        df['Volumen_tendencia'] = df.groupby('Codigo del Cliente (IBS)').apply(
            lambda x: x['Importe FX'].rolling(window=30, min_periods=1).mean() / 
                     x['Importe FX'].rolling(window=90, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error al crear características estadísticas: {str(e)}")
        raise

def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea características de ratio entre variables.
    
    Args:
        df: DataFrame con datos limpios
        
    Returns:
        DataFrame con características de ratio adicionales
    """
    try:
        logger.info("Creando características de ratio")
        
        # Ratios de volumen
        df['Ratio_volumen_diario_promedio'] = df['Volumen_diario'] / df['Volumen_promedio_diario']
        df['Ratio_volumen_ponderado'] = df['Volumen_Ponderado_5'] / df['Volumen_promedio_diario']
        
        # Ratios de spread
        df['Ratio_spread_volumen'] = df['Spread_TC'] * df['Importe FX']
        
        # Ratios de tendencia
        df['Ratio_tendencia_volumen'] = df['Volumen_tendencia'] / df['Volumen_promedio_diario']
        
        # Manejar divisiones por cero
        ratio_columns = [
            'Ratio_volumen_diario_promedio',
            'Ratio_volumen_ponderado',
            'Ratio_spread_volumen',
            'Ratio_tendencia_volumen'
        ]
        
        for col in ratio_columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median())
        
        return df
        
    except Exception as e:
        logger.error(f"Error al crear características de ratio: {str(e)}")
        raise

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza todo el proceso de feature engineering.
    
    Args:
        df: DataFrame con datos limpios
        
    Returns:
        DataFrame con todas las características adicionales
    """
    try:
        logger.info("Iniciando feature engineering")
        
        # Crear características temporales
        df = create_temporal_features(df)
        
        # Crear características estadísticas
        df = create_statistical_features(df)
        
        # Crear características de ratio
        df = create_ratio_features(df)
        
        # Guardar datos con características
        output_path = Path('data/processed/features_data.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Datos con características guardados en {output_path}")
        
        # Mostrar resumen
        logger.info("\nResumen de feature engineering:")
        logger.info(f"Total de características: {len(df.columns)}")
        logger.info("\nCaracterísticas numéricas:")
        for col in df.select_dtypes(include=[np.number]).columns:
            logger.info(f"{col}:")
            logger.info(f"  Media: {df[col].mean():.2f}")
            logger.info(f"  Desviación estándar: {df[col].std():.2f}")
            logger.info(f"  Valores nulos: {df[col].isnull().sum()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error en feature engineering: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Cargar datos limpios
        input_path = Path('data/processed/cleaned_data.csv')
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de datos limpios en {input_path}")
        
        df = pd.read_csv(input_path)
        logger.info(f"Datos limpios cargados desde {input_path}")
        
        # Realizar feature engineering
        df_features = engineer_features(df)
        
    except Exception as e:
        logger.error(f"Error en ejecución principal: {str(e)}")
        raise 