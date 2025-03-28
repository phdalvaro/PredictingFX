import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_split.log'),
        logging.StreamHandler()
    ]
)

def load_features(base_path):
    """Cargar datos con características"""
    try:
        file_path = os.path.join(base_path, 'data', 'processed', 'features.csv')
        df = pd.read_csv(file_path)
        df['Fecha de cierre'] = pd.to_datetime(df['Fecha de cierre'])
        logging.info(f"Datos con características cargados exitosamente: {len(df)} filas")
        return df
    except Exception as e:
        logging.error(f"Error al cargar datos con características: {str(e)}")
        raise

def prepare_volume_prediction_data(df):
    """Preparar datos para predicción de volumen"""
    try:
        # Agrupar por fecha para obtener volumen diario
        daily_data = df.groupby('Fecha de cierre').agg({
            'Importe FX': 'sum',
            'Operaciones_diarias': 'first',
            'Volumen_promedio_diario': 'first',
            'Volumen_std_diario': 'first',
            'TC_std_diario': 'first',
            'TC_range_diario': 'first',
            'Año': 'first',
            'Mes': 'first',
            'Día': 'first',
            'Día de la semana': 'first',
            'Es fin de semana': 'first',
            'Es fin de mes': 'first',
            'Trimestre': 'first'
        }).reset_index()
        
        # Crear características de volumen histórico
        windows = [7, 14, 30]
        for window in windows:
            daily_data[f'Volumen_rolling_{window}d'] = daily_data['Importe FX'].rolling(window=window).mean()
            daily_data[f'Volumen_std_{window}d'] = daily_data['Importe FX'].rolling(window=window).std()
            daily_data[f'Volumen_trend_{window}d'] = daily_data['Importe FX'].rolling(window=window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0]
            )
        
        # Eliminar filas con valores NaN
        daily_data = daily_data.dropna()
        
        # Definir características y variable objetivo
        feature_cols = [col for col in daily_data.columns if col not in ['Fecha de cierre', 'Importe FX']]
        target_col = 'Importe FX'
        
        logging.info(f"Datos de volumen preparados: {len(daily_data)} filas")
        logging.info(f"Características: {len(feature_cols)}")
        
        return daily_data, feature_cols, target_col
    except Exception as e:
        logging.error(f"Error al preparar datos de volumen: {str(e)}")
        raise

def prepare_next_transaction_data(df):
    """Preparar datos para predicción de próxima transacción"""
    try:
        # Definir características relevantes
        feature_cols = [
            'volumen_promedio', 'volumen_total',
            'total_operaciones', 'tc_promedio',
            'spread_promedio', 'Categoria_Cliente', 'Frecuencia_Cliente',
            'volumen_rolling_7d_mean', 'volumen_rolling_14d_mean',
            'volumen_rolling_30d_mean', 'tc_rolling_7d_mean',
            'tc_rolling_14d_mean', 'tc_rolling_30d_mean',
            'Volumen_diario', 'Volumen_promedio_diario', 'Volumen_std_diario',
            'TC_std_diario', 'TC_range_diario'
        ]
        
        # Definir variables objetivo (solo indicadores binarios)
        target_cols = ['Opera_Proximos_7d', 'Opera_Proximos_14d', 'Opera_Proximos_30d']
        
        # Crear copia del DataFrame
        next_transactions = df.copy()
        
        # Imputar valores nulos en características numéricas
        numeric_cols = [col for col in feature_cols if col not in ['Categoria_Cliente', 'Frecuencia_Cliente']]
        next_transactions[numeric_cols] = next_transactions[numeric_cols].fillna(next_transactions[numeric_cols].mean())
        
        # Eliminar filas con valores nulos en variables objetivo
        next_transactions = next_transactions.dropna(subset=target_cols)
        
        logging.info(f"Datos de próxima transacción preparados: {len(next_transactions)} filas")
        logging.info(f"Características: {len(feature_cols)}")
        
        return next_transactions, feature_cols, target_cols
    except Exception as e:
        logging.error(f"Error al preparar datos de próxima transacción: {str(e)}")
        raise

def split_data_temporal(df, test_size=0.2, val_size=0.1):
    """Dividir datos temporalmente"""
    try:
        # Ordenar por fecha
        df = df.sort_values('Fecha de cierre')
        
        # Calcular índices de división
        n = len(df)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        # Dividir datos
        train_data = df.iloc[:val_idx]
        val_data = df.iloc[val_idx:test_idx]
        test_data = df.iloc[test_idx:]
        
        logging.info(f"División temporal completada:")
        logging.info(f"Entrenamiento: {len(train_data)} filas")
        logging.info(f"Validación: {len(val_data)} filas")
        logging.info(f"Prueba: {len(test_data)} filas")
        
        return train_data, val_data, test_data
    except Exception as e:
        logging.error(f"Error al dividir datos temporalmente: {str(e)}")
        raise

def save_split_data(train_data, val_data, test_data, base_path, prediction_type):
    """Guardar conjuntos de datos divididos"""
    try:
        # Crear directorio si no existe
        output_dir = os.path.join(base_path, 'data', 'processed', 'split')
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar conjuntos de datos
        train_data.to_csv(os.path.join(output_dir, f'{prediction_type}_train.csv'), index=False)
        val_data.to_csv(os.path.join(output_dir, f'{prediction_type}_val.csv'), index=False)
        test_data.to_csv(os.path.join(output_dir, f'{prediction_type}_test.csv'), index=False)
        
        logging.info(f"Conjuntos de datos guardados en: {output_dir}")
    except Exception as e:
        logging.error(f"Error al guardar conjuntos de datos: {str(e)}")
        raise

def main():
    try:
        # Obtener ruta base del proyecto
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Crear directorio de logs si no existe
        os.makedirs('logs', exist_ok=True)
        
        # Cargar datos con características
        logging.info("Iniciando división de datos...")
        df = load_features(base_path)
        
        # Preparar y dividir datos para predicción de volumen
        volume_data, volume_features, volume_target = prepare_volume_prediction_data(df)
        volume_train, volume_val, volume_test = split_data_temporal(volume_data)
        save_split_data(volume_train, volume_val, volume_test, base_path, 'volume')
        
        # Preparar y dividir datos para predicción de próxima transacción
        transaction_data, transaction_features, transaction_targets = prepare_next_transaction_data(df)
        transaction_train, transaction_val, transaction_test = split_data_temporal(transaction_data)
        save_split_data(transaction_train, transaction_val, transaction_test, base_path, 'next_transaction')
        
        # Guardar información de características
        feature_info = {
            'volume': {
                'features': volume_features,
                'targets': [volume_target]
            },
            'next_transaction': {
                'features': transaction_features,
                'targets': transaction_targets
            }
        }
        
        with open(os.path.join(base_path, 'data', 'processed', 'split', 'feature_info.txt'), 'w', encoding='utf-8') as f:
            for prediction_type, info in feature_info.items():
                f.write(f"\n{prediction_type.upper()}:\n")
                f.write("Características:\n")
                for feature in info['features']:
                    f.write(f"- {feature}\n")
                f.write("\nVariables objetivo:\n")
                for target in info['targets']:
                    f.write(f"- {target}\n")
        
        logging.info("Proceso de división de datos completado exitosamente")
        
    except Exception as e:
        logging.error(f"Error en el proceso de división de datos: {str(e)}")
        raise

if __name__ == "__main__":
    main() 