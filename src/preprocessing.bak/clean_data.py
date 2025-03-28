import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_cleaning.log'),
        logging.StreamHandler()
    ]
)

def load_data(base_path):
    """Cargar datos del archivo Excel"""
    try:
        file_path = os.path.join(base_path, 'data', 'raw', 'Fluctua.xlsx')
        # Leer la pestaña OPES, saltando la primera fila
        df = pd.read_excel(file_path, sheet_name='OPES', skiprows=1)
        logging.info(f"Datos cargados exitosamente: {len(df)} filas")
        return df
    except Exception as e:
        logging.error(f"Error al cargar datos: {str(e)}")
        raise

def clean_dates(df):
    """Limpiar y validar columnas de fecha"""
    try:
        # Convertir columna de fecha a datetime
        df['Fecha de cierre'] = pd.to_datetime(df['Fecha de cierre'], errors='coerce')
        
        # Eliminar filas con fechas inválidas
        invalid_dates = df['Fecha de cierre'].isna()
        if invalid_dates.any():
            df = df.dropna(subset=['Fecha de cierre'])
            logging.warning(f"Se eliminaron {invalid_dates.sum()} filas con fechas inválidas")
        
        # Ordenar por fecha
        df = df.sort_values('Fecha de cierre')
        logging.info("Fechas limpiadas y ordenadas correctamente")
        return df
    except Exception as e:
        logging.error(f"Error al limpiar fechas: {str(e)}")
        raise

def clean_numeric_columns(df):
    """Limpiar columnas numéricas"""
    try:
        # Columnas numéricas a limpiar
        numeric_columns = [
            'Importe FX', 'T/C Cerrado', 'PX_MID', 'PX_HIGH', 'PX_LAST',
            'Monto Cliente', 'Equivalente en USD (T/C Cerrado)', 
            'Equivalente en USD (T/C Pool)', 'Monto Contraparte'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                # Convertir a numérico, reemplazando errores con NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Reemplazar valores negativos con NaN
                df.loc[df[col] < 0, col] = np.nan
                
                # Calcular estadísticas antes de la limpieza
                missing_before = df[col].isna().sum()
                
                # Imputar valores faltantes con la mediana
                df[col] = df[col].fillna(df[col].median())
                
                # Calcular estadísticas después de la limpieza
                missing_after = df[col].isna().sum()
                logging.info(f"Columna {col}: {missing_before} valores faltantes imputados")
        
        return df
    except Exception as e:
        logging.error(f"Error al limpiar columnas numéricas: {str(e)}")
        raise

def clean_categorical_columns(df):
    """Limpiar columnas categóricas"""
    try:
        # Columnas categóricas a limpiar
        categorical_columns = [
            'Codigo del Cliente (IBS)', 'Rubro de Negocio', 'PN/PJ',
            'Ambito de Negocio', 'Producto', 'Tipo de Operación',
            'Moneda FX', 'Tipo de Cliente', 'Trader', 'Contraparte'
        ]
        
        for col in categorical_columns:
            if col in df.columns:
                # Convertir a string y limpiar espacios
                df[col] = df[col].astype(str).str.strip()
                
                # Reemplazar valores vacíos o 'nan' con 'Desconocido'
                df[col] = df[col].replace(['', 'nan', 'NaN', 'None'], 'Desconocido')
                
                # Eliminar espacios múltiples
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                
                logging.info(f"Columna {col}: {df[col].nunique()} valores únicos")
        
        return df
    except Exception as e:
        logging.error(f"Error al limpiar columnas categóricas: {str(e)}")
        raise

def remove_duplicates(df):
    """Eliminar duplicados"""
    try:
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_rows = initial_rows - len(df)
        
        if removed_rows > 0:
            logging.warning(f"Se eliminaron {removed_rows} filas duplicadas")
        else:
            logging.info("No se encontraron duplicados")
        
        return df
    except Exception as e:
        logging.error(f"Error al eliminar duplicados: {str(e)}")
        raise

def validate_data(df):
    """Validar datos después de la limpieza"""
    try:
        # Verificar que no hay valores faltantes
        missing_values = df.isnull().sum()
        if missing_values.any():
            logging.warning("Se encontraron valores faltantes después de la limpieza:")
            for col, count in missing_values[missing_values > 0].items():
                logging.warning(f"{col}: {count} valores faltantes")
        
        # Verificar rangos de valores numéricos
        numeric_columns = ['Importe FX', 'T/C Cerrado', 'PX_MID']
        for col in numeric_columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                logging.info(f"Rango de {col}: {min_val:.2f} - {max_val:.2f}")
        
        return df
    except Exception as e:
        logging.error(f"Error al validar datos: {str(e)}")
        raise

def save_cleaned_data(df, base_path):
    """Guardar datos limpios"""
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.join(base_path, 'data', 'processed'), exist_ok=True)
        
        output_path = os.path.join(base_path, 'data', 'processed', 'cleaned_data.csv')
        df.to_csv(output_path, index=False)
        logging.info(f"Datos limpios guardados en: {output_path}")
    except Exception as e:
        logging.error(f"Error al guardar datos limpios: {str(e)}")
        raise

def main():
    try:
        # Obtener ruta base del proyecto
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Crear directorio de logs si no existe
        os.makedirs('logs', exist_ok=True)
        
        # Cargar datos
        logging.info("Iniciando proceso de limpieza de datos...")
        df = load_data(base_path)
        
        # Limpiar fechas
        df = clean_dates(df)
        
        # Limpiar columnas numéricas
        df = clean_numeric_columns(df)
        
        # Limpiar columnas categóricas
        df = clean_categorical_columns(df)
        
        # Eliminar duplicados
        df = remove_duplicates(df)
        
        # Validar datos
        df = validate_data(df)
        
        # Guardar datos limpios
        save_cleaned_data(df, base_path)
        
        logging.info("Proceso de limpieza completado exitosamente")
        
    except Exception as e:
        logging.error(f"Error en el proceso de limpieza: {str(e)}")
        raise

if __name__ == "__main__":
    main() 