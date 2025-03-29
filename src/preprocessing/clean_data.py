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
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_raw_data(file_path: Path) -> pd.DataFrame:
    """
    Carga los datos crudos desde el archivo Excel.
    
    Args:
        file_path: Ruta al archivo Excel
        
    Returns:
        DataFrame con datos crudos
    """
    try:
        logger.info(f"Cargando datos desde {file_path}")
        
        # Leer archivo Excel, saltando la primera fila
        df = pd.read_excel(
            file_path,
            sheet_name='OPES',
            skiprows=1,  # Saltar la primera fila
            engine='openpyxl'
        )
        
        # Eliminar columnas no necesarias
        columns_to_drop = [
            'Unnamed: 49', 'Unnamed: 50', 'Unnamed: 51',
            'bbva', 'pn', 'FACTURA/BOLETA', 'Número', 'Bancos',
            'I BCP', 'I BBVA', 'I IBK', 'I BIF', 'I SCOTI', 'I PICHI',
            'S BCP', 'S BBVA', 'S IBK', 'S BIF', 'S SCOTI', 'S PICHI'
        ]
        df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # Renombrar columnas según la estructura real
        column_mapping = {
            'Codigo del Cliente (IBS)': 'Codigo del Cliente (IBS)',
            'Fecha de cierre': 'Fecha de cierre',
            'Importe FX': 'Importe FX',
            'Spread\n': 'Spread_TC',
            'Tipo de Cliente': 'Categoria_Cliente',
            'RUC / DNI termina en': 'Frecuencia_Cliente'
        }
        df = df.rename(columns=column_mapping)
        
        # Crear columnas derivadas
        df['Volumen_promedio_diario'] = df.groupby('Codigo del Cliente (IBS)')['Importe FX'].transform('mean')
        df['Volumen_Ponderado_5'] = df.groupby('Codigo del Cliente (IBS)')['Importe FX'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        df['Volumen_diario'] = df.groupby(['Codigo del Cliente (IBS)', 'Fecha de cierre'])['Importe FX'].transform('sum')
        
        # Convertir fecha a datetime
        df['Fecha de cierre'] = pd.to_datetime(df['Fecha de cierre'])
        df['Mes del año'] = df['Fecha de cierre'].dt.month
        df['Día de la semana nombre'] = df['Fecha de cierre'].dt.day_name()
        
        # Calcular próxima fecha (7 días después)
        df['Proxima_Fecha'] = df['Fecha de cierre'] + pd.Timedelta(days=7)
        
        logger.info(f"Datos cargados exitosamente. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error al cargar datos: {str(e)}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia los datos del DataFrame.
    
    Args:
        df: DataFrame con datos crudos
        
    Returns:
        DataFrame con datos limpios
    """
    try:
        logger.info("Iniciando limpieza de datos")
        
        # Crear copia para no modificar los datos originales
        df_clean = df.copy()
        
        # Convertir columnas de fecha
        date_columns = ['Fecha de cierre', 'Proxima_Fecha']
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col])
        
        # Manejar valores nulos
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # Calcular mediana por grupo de cliente
            median_by_client = df_clean.groupby('Codigo del Cliente (IBS)')[col].transform('median')
            # Reemplazar valores nulos con la mediana correspondiente
            df_clean[col] = df_clean[col].fillna(median_by_client)
            # Si aún hay nulos, reemplazar con la mediana general
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Manejar valores negativos
        for col in numeric_columns:
            df_clean[col] = df_clean[col].clip(lower=0)
        
        # Manejar outliers usando IQR
        for col in numeric_columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Validar tipos de datos
        expected_types = {
            'Codigo del Cliente (IBS)': str,
            'Importe FX': float,
            'Volumen_promedio_diario': float,
            'Spread_TC': float,
            'Volumen_Ponderado_5': float,
            'Volumen_diario': float,
            'Mes del año': int,
            'Día de la semana nombre': str,
            'Categoria_Cliente': str,
            'Frecuencia_Cliente': int
        }
        
        for col, expected_type in expected_types.items():
            if col in df_clean.columns:
                try:
                    df_clean[col] = df_clean[col].astype(expected_type)
                except Exception as e:
                    logger.error(f"Error al convertir columna {col} a tipo {expected_type}: {str(e)}")
                    raise
        
        # Validar rangos de valores
        validations = {
            'Importe FX': (0, float('inf')),
            'Volumen_promedio_diario': (0, float('inf')),
            'Spread_TC': (0, float('inf')),
            'Volumen_Ponderado_5': (0, float('inf')),
            'Volumen_diario': (0, float('inf')),
            'Mes del año': (1, 12),
            'Frecuencia_Cliente': (1, float('inf'))
        }
        
        for col, (min_val, max_val) in validations.items():
            if col in df_clean.columns:
                invalid_mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
                if invalid_mask.any():
                    logger.warning(f"Valores inválidos encontrados en columna {col}")
                    df_clean.loc[invalid_mask, col] = df_clean[col].median()
        
        # Guardar datos limpios
        output_path = Path('data/processed/cleaned_data.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        logger.info(f"Datos limpios guardados en {output_path}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error en limpieza de datos: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Cargar datos crudos desde Excel
        input_path = Path('data/raw/Fluctua.xlsx')
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo Excel en {input_path}")
        
        # Cargar datos
        df = load_raw_data(input_path)
        logger.info(f"Datos crudos cargados desde {input_path}")
        
        # Limpiar datos
        df_clean = clean_data(df)
        
        # Mostrar resumen
        logger.info("\nResumen de limpieza:")
        logger.info(f"Filas originales: {len(df)}")
        logger.info(f"Filas después de limpieza: {len(df_clean)}")
        logger.info("\nColumnas numéricas:")
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            logger.info(f"{col}:")
            logger.info(f"  Media: {df_clean[col].mean():.2f}")
            logger.info(f"  Desviación estándar: {df_clean[col].std():.2f}")
            logger.info(f"  Valores nulos: {df_clean[col].isnull().sum()}")
        
    except Exception as e:
        logger.error(f"Error en ejecución principal: {str(e)}")
        raise
