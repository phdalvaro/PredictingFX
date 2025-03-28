import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging

# Obtener ruta base del proyecto
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_dir = os.path.join(base_path, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'feature_engineering.log')),
        logging.StreamHandler()
    ]
)

def load_cleaned_data(base_path):
    """Cargar datos limpios"""
    try:
        file_path = os.path.join(base_path, 'data', 'processed', 'cleaned_data.csv')
        df = pd.read_csv(file_path)
        df['Fecha de cierre'] = pd.to_datetime(df['Fecha de cierre'])
        # Asegurar que el código del cliente sea string
        df['Codigo del Cliente (IBS)'] = df['Codigo del Cliente (IBS)'].astype(str)
        logging.info(f"Datos limpios cargados exitosamente: {len(df)} filas")
        return df
    except Exception as e:
        logging.error(f"Error al cargar datos limpios: {str(e)}")
        raise

def create_temporal_features(df):
    """Crear características temporales"""
    try:
        # Características de fecha
        df['Año'] = df['Fecha de cierre'].dt.year
        df['Mes'] = df['Fecha de cierre'].dt.month
        df['Día'] = df['Fecha de cierre'].dt.day
        df['Día de la semana'] = df['Fecha de cierre'].dt.dayofweek
        df['Semana del año'] = df['Fecha de cierre'].dt.isocalendar().week
        df['Es fin de semana'] = df['Día de la semana'].isin([5, 6]).astype(int)
        df['Es fin de mes'] = df['Fecha de cierre'].dt.is_month_end.astype(int)
        
        # Características estacionales
        df['Trimestre'] = df['Fecha de cierre'].dt.quarter
        df['Mes del año'] = df['Fecha de cierre'].dt.strftime('%B')
        df['Día de la semana nombre'] = df['Fecha de cierre'].dt.strftime('%A')
        
        # Nuevas características temporales
        df['Hora'] = df['Fecha de cierre'].dt.hour
        df['Minuto'] = df['Fecha de cierre'].dt.minute
        df['Es horario comercial'] = ((df['Hora'] >= 9) & (df['Hora'] <= 17)).astype(int)
        
        logging.info("Características temporales creadas exitosamente")
        return df
    except Exception as e:
        logging.error(f"Error al crear características temporales: {str(e)}")
        raise

def create_volume_features(df):
    """Crear características de volumen"""
    try:
        # Agrupar por fecha
        daily_stats = df.groupby('Fecha de cierre').agg({
            'Importe FX': ['count', 'sum', 'mean', 'std']
        }).reset_index()
        daily_stats.columns = ['Fecha de cierre', 'Operaciones_diarias', 'Volumen_diario', 'Volumen_promedio_diario', 'Volumen_std_diario']
        
        # Unir con el DataFrame original
        df = df.merge(daily_stats, on='Fecha de cierre', how='left')
        
        # Calcular características rolling
        windows = [7, 14, 30]
        for window in windows:
            # Ordenar por fecha para cálculos rolling
            temp_df = df.sort_values('Fecha de cierre')
            
            # Volumen rolling por cliente
            volume_rolling = temp_df.groupby('Codigo del Cliente (IBS)')['Importe FX'].rolling(window=window).agg(['mean', 'std', 'sum']).reset_index()
            volume_rolling.columns = ['index', 'Codigo del Cliente (IBS)', f'volumen_rolling_{window}d_mean', f'volumen_rolling_{window}d_std', f'volumen_rolling_{window}d_sum']
            volume_rolling = volume_rolling.drop('index', axis=1)
            
            # Asegurar que el código del cliente sea string
            volume_rolling['Codigo del Cliente (IBS)'] = volume_rolling['Codigo del Cliente (IBS)'].astype(str)
            
            # Unir con el DataFrame principal
            df = df.merge(volume_rolling, on='Codigo del Cliente (IBS)', how='left')
        
        logging.info("Características de volumen creadas exitosamente")
        return df
    except Exception as e:
        logging.error(f"Error al crear características de volumen: {str(e)}")
        raise

def create_market_features(df):
    """Crear características de mercado"""
    try:
        # Asegurar que las columnas numéricas sean float
        numeric_cols = ['T/C Cerrado', 'PX_MID']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Spread entre tipos de cambio
        df['Spread_TC'] = df['T/C Cerrado'] - df['PX_MID']
        
        # Volatilidad diaria del tipo de cambio
        tc_daily_stats = df.groupby('Fecha de cierre').agg({
            'T/C Cerrado': ['std', lambda x: x.max() - x.min()]
        }).reset_index()
        tc_daily_stats.columns = ['Fecha de cierre', 'TC_std_diario', 'TC_range_diario']
        
        # Asegurar que la fecha sea datetime
        tc_daily_stats['Fecha de cierre'] = pd.to_datetime(tc_daily_stats['Fecha de cierre'])
        
        # Unir con el DataFrame original
        df = df.merge(tc_daily_stats, on='Fecha de cierre', how='left')
        
        # Calcular características rolling del tipo de cambio
        windows = [7, 14, 30]
        for window in windows:
            # Ordenar por fecha para cálculos rolling
            temp_df = df.sort_values('Fecha de cierre')
            
            # Tipo de cambio rolling
            tc_rolling = temp_df.groupby('Fecha de cierre')['T/C Cerrado'].rolling(window=window).agg(['mean', 'std']).reset_index()
            tc_rolling.columns = ['index', 'Fecha de cierre', f'tc_rolling_{window}d_mean', f'tc_rolling_{window}d_std']
            tc_rolling = tc_rolling.drop('index', axis=1)
            
            # Asegurar que la fecha sea datetime
            tc_rolling['Fecha de cierre'] = pd.to_datetime(tc_rolling['Fecha de cierre'])
            
            # Unir con el DataFrame principal
            df = df.merge(tc_rolling, on='Fecha de cierre', how='left')
        
        # Nuevas características de mercado
        df['Volatilidad_Horaria'] = df.groupby(['Fecha de cierre', 'Hora'])['T/C Cerrado'].transform('std')
        df['Rango_Horario'] = df.groupby(['Fecha de cierre', 'Hora'])['T/C Cerrado'].transform(lambda x: x.max() - x.min())
        
        logging.info("Características de mercado creadas exitosamente")
        return df
    except Exception as e:
        logging.error(f"Error al crear características de mercado: {str(e)}")
        raise

def create_client_features(df):
    """Crear características de cliente"""
    try:
        # Asegurar que el código del cliente sea string
        df['Codigo del Cliente (IBS)'] = df['Codigo del Cliente (IBS)'].astype(str)
        
        # Estadísticas históricas por cliente
        client_stats = df.groupby('Codigo del Cliente (IBS)').agg({
            'Importe FX': ['count', 'mean', 'std', 'sum'],
            'T/C Cerrado': ['mean', 'std'],
            'Spread_TC': 'mean'
        }).reset_index()
        
        client_stats.columns = [
            'Codigo del Cliente (IBS)',
            'total_operaciones', 'volumen_promedio', 'volumen_std', 'volumen_total',
            'tc_promedio', 'tc_std', 'spread_promedio'
        ]
        
        # Unir con el DataFrame original
        df = df.merge(client_stats, on='Codigo del Cliente (IBS)', how='left')
        
        # Crear variables categóricas
        df['Categoria_Cliente'] = pd.qcut(df['volumen_total'], q=5, labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto'])
        df['Frecuencia_Cliente'] = pd.qcut(df['total_operaciones'], q=5, labels=['Muy Baja', 'Baja', 'Media', 'Alta', 'Muy Alta'])
        
        logging.info("Características de cliente creadas exitosamente")
        return df
    except Exception as e:
        logging.error(f"Error al crear características de cliente: {str(e)}")
        raise

def create_target_variables(df):
    """Crear variables objetivo"""
    try:
        # Ordenar por cliente y fecha
        df = df.sort_values(['Codigo del Cliente (IBS)', 'Fecha de cierre'])
        
        # 1. Próxima fecha de transacción
        df['Proxima_Fecha'] = df.groupby('Codigo del Cliente (IBS)')['Fecha de cierre'].shift(-1)
        df['Dias_Proxima_Operacion'] = (df['Proxima_Fecha'] - df['Fecha de cierre']).dt.days
        
        # 2. Volumen de próxima transacción
        df['Proximo_Volumen'] = df.groupby('Codigo del Cliente (IBS)')['Importe FX'].shift(-1)
        
        # 3. Indicador de transacción en próximos X días
        for dias in [7, 14, 30]:
            df[f'Opera_Proximos_{dias}d'] = (df['Dias_Proxima_Operacion'] <= dias).astype(int)
        
        logging.info("Variables objetivo creadas exitosamente")
        return df
    except Exception as e:
        logging.error(f"Error al crear variables objetivo: {str(e)}")
        raise

def save_features(df, base_path):
    """Guardar datos con nuevas características"""
    try:
        # Crear directorio si no existe
        output_dir = os.path.join(base_path, 'data', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar archivo
        output_path = os.path.join(output_dir, 'features.csv')
        df.to_csv(output_path, index=False)
        
        # Guardar diccionario de características
        feature_dict = {
            'Temporales': [col for col in df.columns if col.startswith(('Año', 'Mes', 'Día', 'Trimestre', 'Es_'))],
            'Volumen': [col for col in df.columns if 'volumen' in col.lower()],
            'Mercado': [col for col in df.columns if any(x in col for x in ['TC_', 'Spread_', 'tc_rolling'])],
            'Cliente': [col for col in df.columns if any(x in col for x in ['total_', 'Categoria_', 'Frecuencia_'])],
            'Target': [col for col in df.columns if any(x in col for x in ['Proxima_', 'Opera_', 'Proximo_'])]
        }
        
        with open(os.path.join(output_dir, 'feature_dictionary.txt'), 'w', encoding='utf-8') as f:
            for category, features in feature_dict.items():
                f.write(f"\n{category}:\n")
                for feature in features:
                    f.write(f"- {feature}\n")
        
        logging.info(f"Datos con características guardados en: {output_path}")
        logging.info(f"Diccionario de características guardado en: {os.path.join(output_dir, 'feature_dictionary.txt')}")
    except Exception as e:
        logging.error(f"Error al guardar características: {str(e)}")
        raise

def calculate_weighted_metrics(df):
    """Calcular métricas ponderadas de las últimas 5 transacciones por ID"""
    try:
        # Ordenar datos por ID y fecha
        df_sorted = df.sort_values(['Codigo del Cliente (IBS)', 'Fecha de cierre'])
        
        # Calcular spread
        df_sorted['Spread_TC'] = df_sorted['T/C Cerrado'] - df_sorted['PX_MID']
        
        # Calcular volumen diario por cliente
        daily_volume = df_sorted.groupby(['Codigo del Cliente (IBS)', 'Fecha de cierre'])['Importe FX'].sum().reset_index()
        df_sorted = df_sorted.merge(daily_volume, on=['Codigo del Cliente (IBS)', 'Fecha de cierre'], suffixes=('', '_daily'))
        df_sorted['Volumen_diario'] = df_sorted['Importe FX_daily']
        df_sorted = df_sorted.drop('Importe FX_daily', axis=1)
        
        # Calcular pesos (más reciente = mayor peso)
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        
        # Inicializar columnas para métricas ponderadas
        df_sorted['Volumen_Ponderado_5'] = np.nan
        df_sorted['Spread_Ponderado_5'] = np.nan
        
        # Calcular métricas ponderadas para cada ID
        for client_id in df_sorted['Codigo del Cliente (IBS)'].unique():
            # Obtener datos del cliente
            client_data = df_sorted[df_sorted['Codigo del Cliente (IBS)'] == client_id].copy()
            
            # Calcular métricas ponderadas
            for i in range(len(client_data)):
                if i >= 4:  # Solo si hay al menos 5 transacciones previas
                    # Obtener últimas 5 transacciones
                    last_5_volume = client_data['Importe FX'].iloc[i-4:i+1].values
                    last_5_spread = client_data['Spread_TC'].iloc[i-4:i+1].values
                    
                    # Calcular promedios ponderados
                    df_sorted.loc[client_data.index[i], 'Volumen_Ponderado_5'] = np.average(last_5_volume, weights=weights)
                    df_sorted.loc[client_data.index[i], 'Spread_Ponderado_5'] = np.average(last_5_spread, weights=weights)
        
        logging.info("Métricas ponderadas calculadas exitosamente")
        return df_sorted
    except Exception as e:
        logging.error(f"Error al calcular métricas ponderadas: {str(e)}")
        raise

def main():
    """Función principal"""
    try:
        # Obtener ruta base del proyecto
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Cargar datos limpios
        df = load_cleaned_data(base_path)
        
        # Crear características
        df = create_temporal_features(df)
        df = create_volume_features(df)
        df = create_market_features(df)
        df = create_client_features(df)
        df = create_target_variables(df)
        df = calculate_weighted_metrics(df)
        
        # Guardar características
        save_features(df, base_path)
        
        logging.info("Proceso de ingeniería de características completado exitosamente")
    except Exception as e:
        logging.error(f"Error en el proceso principal: {str(e)}")
        raise

if __name__ == "__main__":
    main() 