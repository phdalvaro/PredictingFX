import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
from prophet_model import ProphetForecaster
import joblib
from prophet import Prophet
from xgboost import XGBRegressor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_comparison.log'),
        logging.StreamHandler()
    ]
)

def load_data():
    """Carga los datos de entrenamiento y prueba"""
    try:
        # Cargar datos procesados
        data_path = os.path.join('data', 'processed', 'features.csv')
        df = pd.read_csv(data_path)
        
        # Verificar columnas disponibles
        logging.info(f"Columnas disponibles: {df.columns.tolist()}")
        
        # Asegurarse de que tenemos las columnas necesarias
        required_columns = ['Fecha de cierre', 'Volumen_diario']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Faltan columnas requeridas. Necesitamos: {required_columns}")
        
        # Renombrar columnas para consistencia
        df = df.rename(columns={
            'Fecha de cierre': 'Fecha',
            'Volumen_diario': 'Volumen'
        })
        
        # Convertir columna de fecha
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Dividir datos en entrenamiento y prueba (80-20)
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        return train_df, test_df
    except Exception as e:
        logging.error(f"Error al cargar datos: {str(e)}")
        raise

def evaluate_model(y_true, y_pred, model_name):
    """
    Evalúa el modelo usando diferentes métricas.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        model_name: Nombre del modelo para logging
    """
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        logging.info(f"\nMétricas para {model_name}:")
        logging.info(f"RMSE: {rmse:.2f}")
        logging.info(f"MAE: {mae:.2f}")
        logging.info(f"R²: {r2:.4f}")
        
        return rmse, mae, r2
    except Exception as e:
        logging.error(f"Error al evaluar modelo {model_name}: {str(e)}")
        raise

def plot_predictions(test_df, prophet_pred, xgboost_pred, save_path):
    """
    Genera gráfico comparativo de predicciones.
    
    Args:
        test_df: DataFrame con datos de prueba
        prophet_pred: Predicciones de Prophet
        xgboost_pred: Predicciones de XGBoost
        save_path: Ruta para guardar el gráfico
    """
    try:
        plt.figure(figsize=(15, 7))
        plt.plot(test_df['Fecha'], test_df['Volumen'], label='Real', color='black')
        plt.plot(test_df['Fecha'], prophet_pred, label='Prophet', color='blue')
        plt.plot(test_df['Fecha'], xgboost_pred, label='XGBoost', color='red')
        
        plt.title('Comparación de Predicciones: Prophet vs XGBoost', fontsize=14)
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Volumen', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()
        
        logging.info(f"Gráfico guardado en {save_path}")
    except Exception as e:
        logging.error(f"Error al generar gráfico: {str(e)}")
        raise

def prepare_data_for_xgboost(df):
    """
    Prepara los datos para XGBoost.
    
    Args:
        df: DataFrame con los datos
        
    Returns:
        DataFrame preparado para XGBoost
    """
    # Crear una copia del DataFrame
    df_xgb = df.copy()
    
    # Convertir columnas categóricas a numéricas
    categorical_columns = [
        'Codigo del Cliente (IBS)',
        'Mes del año',
        'Día de la semana nombre',
        'Categoria_Cliente',
        'Frecuencia_Cliente',
        'Proxima_Fecha'
    ]
    
    # Eliminar columnas que no son necesarias para el modelo
    df_xgb = df_xgb.drop(categorical_columns + ['Fecha de cierre', 'Volumen_diario'], axis=1, errors='ignore')
    
    # Manejar valores faltantes en todas las columnas numéricas
    numeric_columns = df_xgb.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        if df_xgb[col].isnull().any():
            # Calcular la media excluyendo valores nulos
            mean_value = df_xgb[col].mean()
            # Rellenar valores nulos con la media
            df_xgb[col] = df_xgb[col].fillna(mean_value)
            # Si aún hay valores nulos, usar forward fill y backward fill
            if df_xgb[col].isnull().any():
                df_xgb[col] = df_xgb[col].ffill().bfill()
                # Si aún hay valores nulos después de todo, usar 0
                df_xgb[col] = df_xgb[col].fillna(0)
    
    return df_xgb

def prepare_data_for_prophet(df):
    """
    Prepara los datos para Prophet.
    
    Args:
        df: DataFrame con los datos
        
    Returns:
        DataFrame preparado para Prophet
    """
    # Crear una copia del DataFrame
    prophet_df = df.copy()
    
    # Preparar columnas para Prophet
    prophet_df['ds'] = prophet_df['Fecha de cierre']
    prophet_df['y'] = prophet_df['Volumen_diario']
    
    # Lista de regresores que queremos usar
    regressors = [
        'volumen_rolling_7d_mean',
        'volumen_rolling_30d_mean',
        'Spread_TC',
        'total_operaciones',
        'Volumen_Ponderado_5'
    ]
    
    # Manejar valores faltantes en los regresores
    for regressor in regressors:
        if regressor in prophet_df.columns:
            if prophet_df[regressor].isnull().any():
                # Calcular la media excluyendo valores nulos
                mean_value = prophet_df[regressor].mean()
                # Rellenar valores nulos con la media
                prophet_df[regressor] = prophet_df[regressor].fillna(mean_value)
                # Si aún hay valores nulos, usar forward fill y backward fill
                if prophet_df[regressor].isnull().any():
                    prophet_df[regressor] = prophet_df[regressor].ffill().bfill()
                    # Si aún hay valores nulos después de todo, usar 0
                    prophet_df[regressor] = prophet_df[regressor].fillna(0)
        else:
            logging.warning(f"Regresor {regressor} no encontrado en los datos. Se omitirá.")
            regressors.remove(regressor)
    
    # Eliminar columnas que no son regresores ni ds/y
    columns_to_keep = ['ds', 'y'] + regressors
    prophet_df = prophet_df[columns_to_keep]
    
    # Verificar que no haya valores nulos en el DataFrame final
    null_columns = prophet_df.columns[prophet_df.isnull().any()].tolist()
    if null_columns:
        raise ValueError(f"Aún hay valores nulos en las columnas: {null_columns}")
    
    return prophet_df, regressors

def main():
    """
    Función principal que ejecuta la comparación de modelos.
    """
    try:
        # Cargar datos
        df = pd.read_csv('data/processed/features.csv')
        logging.info("Columnas disponibles: %s", df.columns.tolist())
        
        # Convertir la columna de fecha a datetime
        df['Fecha de cierre'] = pd.to_datetime(df['Fecha de cierre'])
        
        # Verificar el rango de fechas
        min_date = df['Fecha de cierre'].min()
        max_date = df['Fecha de cierre'].max()
        logging.info("Rango de fechas: %s a %s", min_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d'))
        
        # Calcular el punto de división (80% entrenamiento, 20% prueba)
        split_point = df['Fecha de cierre'].quantile(0.8)
        logging.info("Punto de división: %s", split_point.strftime('%Y-%m-%d'))
        
        # Dividir datos en entrenamiento y prueba
        train_df = df[df['Fecha de cierre'] <= split_point]
        test_df = df[df['Fecha de cierre'] > split_point]
        
        logging.info("Tamaño del conjunto de entrenamiento: %d", len(train_df))
        logging.info("Tamaño del conjunto de prueba: %d", len(test_df))
        
        # Preparar datos para Prophet
        train_prophet_df, regressors = prepare_data_for_prophet(train_df)
        test_prophet_df, _ = prepare_data_for_prophet(test_df)
        
        # Entrenar modelo Prophet
        logging.info("Entrenando modelo Prophet...")
        prophet = Prophet()
        
        # Agregar regresores al modelo
        for regressor in regressors:
            prophet.add_regressor(regressor)
        
        prophet.fit(train_prophet_df)
        logging.info("Modelo Prophet entrenado exitosamente")
        
        # Generar pronósticos con Prophet
        future_df = test_prophet_df[['ds'] + regressors]
        prophet_pred = prophet.predict(future_df)['yhat'].values
        
        # Preparar datos para XGBoost
        X_train = prepare_data_for_xgboost(train_df)
        y_train = train_df['Volumen_diario']
        X_test = prepare_data_for_xgboost(test_df)
        y_test = test_df['Volumen_diario']
        
        # Entrenar modelo XGBoost
        logging.info("Entrenando modelo XGBoost...")
        xgb_model = XGBRegressor(random_state=42)
        xgb_model.fit(X_train, y_train)
        logging.info("Modelo XGBoost entrenado exitosamente")
        
        # Generar pronósticos con XGBoost
        xgb_pred = xgb_model.predict(X_test)
        
        # Calcular métricas
        prophet_rmse = np.sqrt(mean_squared_error(y_test, prophet_pred))
        prophet_mae = mean_absolute_error(y_test, prophet_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        
        # Crear DataFrame con resultados
        results = pd.DataFrame({
            'Fecha': test_df['Fecha de cierre'],
            'Real': y_test,
            'Prophet': prophet_pred,
            'XGBoost': xgb_pred
        })
        
        # Crear directorio de resultados si no existe
        os.makedirs('results', exist_ok=True)
        
        # Guardar resultados
        results.to_csv('results/model_comparison.csv', index=False)
        logging.info("Resultados guardados en results/model_comparison.csv")
        
        # Imprimir métricas
        logging.info("\nMétricas de evaluación:")
        logging.info("Prophet - RMSE: %.2f, MAE: %.2f", prophet_rmse, prophet_mae)
        logging.info("XGBoost - RMSE: %.2f, MAE: %.2f", xgb_rmse, xgb_mae)
        
    except Exception as e:
        logging.error("Error en el proceso principal: %s", str(e))
        raise

if __name__ == "__main__":
    main() 