import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os
from datetime import datetime, timedelta
import joblib
import json
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_monitoring.log'),
        logging.StreamHandler()
    ]
)

class ModelMonitor:
    def __init__(self, model_path, threshold_rmse=1000, threshold_mae=500):
        """
        Inicializa el monitor del modelo.
        
        Args:
            model_path: Ruta al modelo guardado
            threshold_rmse: Umbral de alerta para RMSE
            threshold_mae: Umbral de alerta para MAE
        """
        self.model_path = model_path
        self.threshold_rmse = threshold_rmse
        self.threshold_mae = threshold_mae
        self.model = joblib.load(model_path)
        self.metrics_history = []
        
    def load_recent_data(self, days=30):
        """
        Carga los datos más recientes para evaluación.
        
        Args:
            days: Número de días de datos a cargar
            
        Returns:
            DataFrame con los datos recientes
        """
        try:
            # Cargar datos procesados
            data_path = os.path.join('data', 'processed', 'features.csv')
            df = pd.read_csv(data_path)
            
            # Convertir columna de fecha
            df['Fecha de cierre'] = pd.to_datetime(df['Fecha de cierre'])
            
            # Filtrar datos recientes
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_data = df[df['Fecha de cierre'] >= cutoff_date]
            
            return recent_data
        except Exception as e:
            logging.error(f"Error al cargar datos recientes: {str(e)}")
            raise
    
    def prepare_data(self, df):
        """
        Prepara los datos para el modelo.
        
        Args:
            df: DataFrame con los datos
            
        Returns:
            DataFrame preparado para el modelo
        """
        try:
            # Crear features de interacción
            df['Importe_FX_Volumen_Promedio'] = df['Importe FX'] * df['Volumen_promedio_diario']
            df['Importe_FX_Spread'] = df['Importe FX'] * df['Spread_TC']
            df['Volumen_Ponderado_Spread'] = df['Volumen_Ponderado_5'] * df['Spread_TC']
            df['Volumen_Importe_Ratio'] = df['Volumen_diario'] / df['Importe FX']
            df['Spread_Volumen_Ratio'] = df['Spread_TC'] / df['Volumen_diario']
            
            # Eliminar columnas no necesarias
            categorical_columns = [
                'Codigo del Cliente (IBS)',
                'Mes del año',
                'Día de la semana nombre',
                'Categoria_Cliente',
                'Frecuencia_Cliente',
                'Proxima_Fecha'
            ]
            
            df = df.drop(categorical_columns + ['Fecha de cierre', 'Volumen_diario'], axis=1, errors='ignore')
            
            # Normalizar features
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())
                    df[col] = df[col].ffill().bfill()
                    df[col] = df[col].fillna(0)
            
            return df
        except Exception as e:
            logging.error(f"Error al preparar datos: {str(e)}")
            raise
    
    def evaluate_model(self, X, y_true):
        """
        Evalúa el modelo en los datos proporcionados.
        
        Args:
            X: Features
            y_true: Valores reales
            
        Returns:
            Dict con métricas de evaluación
        """
        try:
            y_pred = self.model.predict(X)
            
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'timestamp': datetime.now().isoformat()
            }
            
            return metrics
        except Exception as e:
            logging.error(f"Error al evaluar modelo: {str(e)}")
            raise
    
    def check_alerts(self, metrics):
        """
        Verifica si hay alertas basadas en las métricas.
        
        Args:
            metrics: Dict con métricas de evaluación
            
        Returns:
            Lista de alertas
        """
        alerts = []
        
        if metrics['rmse'] > self.threshold_rmse:
            alerts.append(f"RMSE ({metrics['rmse']:.2f}) excede el umbral ({self.threshold_rmse})")
        
        if metrics['mae'] > self.threshold_mae:
            alerts.append(f"MAE ({metrics['mae']:.2f}) excede el umbral ({self.threshold_mae})")
        
        if metrics['r2'] < 0.9:
            alerts.append(f"R² ({metrics['r2']:.4f}) está por debajo del umbral (0.9)")
        
        return alerts
    
    def save_metrics(self, metrics):
        """
        Guarda las métricas en el historial.
        
        Args:
            metrics: Dict con métricas de evaluación
        """
        try:
            self.metrics_history.append(metrics)
            
            # Guardar en archivo
            metrics_path = Path('results/metrics_history.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
            
            logging.info(f"Métricas guardadas en {metrics_path}")
        except Exception as e:
            logging.error(f"Error al guardar métricas: {str(e)}")
            raise
    
    def run_monitoring(self):
        """
        Ejecuta el proceso de monitoreo completo.
        """
        try:
            # Cargar datos recientes
            recent_data = self.load_recent_data()
            
            # Preparar datos
            X = self.prepare_data(recent_data)
            y_true = recent_data['Volumen_diario']
            
            # Evaluar modelo
            metrics = self.evaluate_model(X, y_true)
            
            # Verificar alertas
            alerts = self.check_alerts(metrics)
            
            # Guardar métricas
            self.save_metrics(metrics)
            
            # Logging de resultados
            logging.info("\nResultados del monitoreo:")
            logging.info(f"RMSE: {metrics['rmse']:.2f}")
            logging.info(f"MAE: {metrics['mae']:.2f}")
            logging.info(f"R²: {metrics['r2']:.4f}")
            
            if alerts:
                logging.warning("\nAlertas detectadas:")
                for alert in alerts:
                    logging.warning(alert)
            else:
                logging.info("\nNo se detectaron alertas")
            
        except Exception as e:
            logging.error(f"Error en el proceso de monitoreo: {str(e)}")
            raise

def main():
    """
    Función principal que ejecuta el monitoreo del modelo.
    """
    try:
        # Crear directorio de logs si no existe
        os.makedirs('logs', exist_ok=True)
        
        # Inicializar monitor
        monitor = ModelMonitor('models/xgboost_model.joblib')
        
        # Ejecutar monitoreo
        monitor.run_monitoring()
        
    except Exception as e:
        logging.error(f"Error en el proceso principal: {str(e)}")
        raise

if __name__ == "__main__":
    main() 