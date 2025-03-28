import pandas as pd
import numpy as np
from prophet import Prophet
import logging
import os
from datetime import datetime
import joblib

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/prophet_model.log'),
        logging.StreamHandler()
    ]
)

class ProphetForecaster:
    def __init__(self, changepoint_prior_scale=0.05, seasonality_prior_scale=10.0):
        """
        Inicializa el modelo Prophet.
        
        Args:
            changepoint_prior_scale (float): Flexibilidad del modelo para cambios de tendencia
            seasonality_prior_scale (float): Flexibilidad del modelo para estacionalidad
        """
        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative',  # Mejor para datos financieros
            growth='linear'  # Tendencia lineal
        )
        
        # Agregar regresores adicionales
        self.model.add_regressor('Volumen_promedio_diario')
        self.model.add_regressor('volumen_rolling_7d_mean')
        self.model.add_regressor('volumen_rolling_30d_mean')
        self.model.add_regressor('Spread_TC')
        self.model.add_regressor('total_operaciones')
        
    def prepare_data(self, df):
        """
        Prepara los datos para Prophet.
        
        Args:
            df (pd.DataFrame): DataFrame con columnas 'Fecha' y 'Volumen'
            
        Returns:
            pd.DataFrame: DataFrame preparado para Prophet
        """
        # Prophet requiere columnas 'ds' y 'y'
        prophet_df = df.copy()
        prophet_df['ds'] = prophet_df['Fecha']
        prophet_df['y'] = prophet_df['Volumen']
        
        # Agregar regresores
        for regressor in ['Volumen_promedio_diario', 'volumen_rolling_7d_mean', 
                         'volumen_rolling_30d_mean', 'Spread_TC', 'total_operaciones']:
            if regressor in df.columns:
                prophet_df[regressor] = df[regressor]
        
        return prophet_df
    
    def fit(self, df):
        """
        Entrena el modelo Prophet.
        
        Args:
            df (pd.DataFrame): DataFrame con datos de entrenamiento
        """
        try:
            prophet_df = self.prepare_data(df)
            logging.info("Entrenando modelo Prophet...")
            self.model.fit(prophet_df)
            logging.info("Modelo Prophet entrenado exitosamente")
        except Exception as e:
            logging.error(f"Error al entrenar el modelo Prophet: {str(e)}")
            raise
    
    def predict(self, periods=30):
        """
        Genera pronósticos con Prophet.
        
        Args:
            periods (int): Número de períodos a pronosticar
            
        Returns:
            pd.DataFrame: DataFrame con pronósticos
        """
        try:
            future = self.model.make_future_dataframe(periods=periods)
            forecast = self.model.predict(future)
            return forecast
        except Exception as e:
            logging.error(f"Error al generar pronósticos con Prophet: {str(e)}")
            raise
    
    def save_model(self, path):
        """
        Guarda el modelo entrenado.
        
        Args:
            path (str): Ruta donde guardar el modelo
        """
        try:
            joblib.dump(self.model, path)
            logging.info(f"Modelo guardado en {path}")
        except Exception as e:
            logging.error(f"Error al guardar el modelo: {str(e)}")
            raise
    
    def load_model(self, path):
        """
        Carga un modelo entrenado.
        
        Args:
            path (str): Ruta del modelo a cargar
        """
        try:
            self.model = joblib.load(path)
            logging.info(f"Modelo cargado desde {path}")
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {str(e)}")
            raise

def main():
    """Función principal para entrenar y evaluar el modelo Prophet"""
    try:
        # Cargar datos
        data_path = os.path.join('data', 'processed', 'features.csv')
        df = pd.read_csv(data_path)
        
        # Convertir columna de fecha
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Crear y entrenar modelo
        forecaster = ProphetForecaster()
        forecaster.fit(df)
        
        # Generar pronósticos
        forecast = forecaster.predict(periods=30)
        
        # Guardar modelo
        model_path = os.path.join('models', 'prophet_model.joblib')
        forecaster.save_model(model_path)
        
        # Guardar pronósticos
        forecast_path = os.path.join('data', 'processed', 'forecasts', 'prophet_forecast.csv')
        forecast.to_csv(forecast_path, index=False)
        
        logging.info("Proceso completado exitosamente")
        
    except Exception as e:
        logging.error(f"Error en el proceso principal: {str(e)}")
        raise

if __name__ == "__main__":
    main() 