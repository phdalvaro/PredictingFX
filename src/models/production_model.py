import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import hashlib
import secrets
from cryptography.fernet import Fernet
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecurityManager:
    """Manejador de seguridad para el modelo."""
    
    def __init__(self, key_file: str = 'security/key.key'):
        """
        Inicializa el manejador de seguridad.
        
        Args:
            key_file: Ruta al archivo de clave
        """
        self.key_file = key_file
        self.key = self._load_or_generate_key()
        self.cipher_suite = Fernet(self.key)
        self._setup_security()
    
    def _setup_security(self) -> None:
        """Configura la seguridad inicial del sistema."""
        try:
            # Crear directorios de seguridad si no existen
            os.makedirs('security', exist_ok=True)
            os.makedirs('logs/security', exist_ok=True)
            
            # Configurar permisos de archivos
            if os.path.exists(self.key_file):
                os.chmod(self.key_file, 0o600)  # Solo lectura/escritura para el propietario
            
            # Verificar integridad de archivos críticos
            self._verify_critical_files()
            
            logging.info("Configuración de seguridad completada")
        except Exception as e:
            logging.error(f"Error en configuración de seguridad: {str(e)}")
            raise
    
    def _verify_critical_files(self) -> None:
        """Verifica la integridad de archivos críticos."""
        critical_files = [
            'models/xgboost_model.joblib',
            'models/scaler.joblib',
            'security/key.key'
        ]
        
        for file in critical_files:
            if not os.path.exists(file):
                raise SecurityError(f"Archivo crítico no encontrado: {file}")
            
            # Verificar permisos
            if os.path.getmode(file) & 0o777 != 0o600:
                logging.warning(f"Permisos inseguros en archivo: {file}")
                os.chmod(file, 0o600)
    
    def _load_or_generate_key(self) -> bytes:
        """
        Carga o genera una clave de encriptación.
        
        Returns:
            bytes: Clave de encriptación
        """
        try:
            if os.path.exists(self.key_file):
                with open(self.key_file, 'rb') as f:
                    key = f.read()
                    if not self._validate_key(key):
                        raise SecurityError("Clave de encriptación inválida")
                    return key
            else:
                key = Fernet.generate_key()
                os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                os.chmod(self.key_file, 0o600)
                return key
        except Exception as e:
            logging.error(f"Error al manejar clave de seguridad: {str(e)}")
            raise
    
    def _validate_key(self, key: bytes) -> bool:
        """
        Valida la clave de encriptación.
        
        Args:
            key: Clave a validar
            
        Returns:
            bool: True si la clave es válida
        """
        try:
            # Verificar longitud
            if len(key) != 44:  # Longitud estándar de Fernet
                return False
            
            # Verificar formato base64
            try:
                base64.b64decode(key)
                return True
            except:
                return False
        except Exception as e:
            logging.error(f"Error al validar clave: {str(e)}")
            return False
    
    def encrypt_data(self, data: str) -> str:
        """
        Encripta datos sensibles.
        
        Args:
            data: Datos a encriptar
            
        Returns:
            str: Datos encriptados
        """
        try:
            # Validar datos de entrada
            if not isinstance(data, str):
                raise ValueError("Los datos deben ser una cadena de texto")
            
            # Generar IV único
            iv = os.urandom(16)
            
            # Encriptar datos
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            
            # Combinar IV y datos encriptados
            combined = base64.b64encode(iv + encrypted_data).decode()
            
            # Registrar intento de encriptación
            self._log_security_event('encrypt', 'success')
            
            return combined
        except Exception as e:
            self._log_security_event('encrypt', 'failure', str(e))
            logging.error(f"Error al encriptar datos: {str(e)}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Desencripta datos sensibles.
        
        Args:
            encrypted_data: Datos encriptados
            
        Returns:
            str: Datos desencriptados
        """
        try:
            # Validar datos de entrada
            if not isinstance(encrypted_data, str):
                raise ValueError("Los datos encriptados deben ser una cadena de texto")
            
            # Decodificar datos combinados
            combined = base64.b64decode(encrypted_data.encode())
            
            # Separar IV y datos encriptados
            iv = combined[:16]
            encrypted = combined[16:]
            
            # Desencriptar datos
            decrypted_data = self.cipher_suite.decrypt(encrypted)
            
            # Registrar intento de desencriptación
            self._log_security_event('decrypt', 'success')
            
            return decrypted_data.decode()
        except Exception as e:
            self._log_security_event('decrypt', 'failure', str(e))
            logging.error(f"Error al desencriptar datos: {str(e)}")
            raise
    
    def generate_hash(self, data: str) -> str:
        """
        Genera un hash seguro de los datos.
        
        Args:
            data: Datos a hashear
            
        Returns:
            str: Hash de los datos
        """
        try:
            # Validar datos de entrada
            if not isinstance(data, str):
                raise ValueError("Los datos deben ser una cadena de texto")
            
            # Generar salt único
            salt = os.urandom(16)
            
            # Crear KDF
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            # Generar hash
            key = base64.b64encode(kdf.derive(data.encode())).decode()
            
            # Registrar generación de hash
            self._log_security_event('hash', 'success')
            
            return key
        except Exception as e:
            self._log_security_event('hash', 'failure', str(e))
            logging.error(f"Error al generar hash: {str(e)}")
            raise
    
    def _log_security_event(self, event_type: str, status: str, details: str = None) -> None:
        """
        Registra eventos de seguridad.
        
        Args:
            event_type: Tipo de evento
            status: Estado del evento
            details: Detalles adicionales
        """
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'event_type': event_type,
                'status': status,
                'details': details
            }
            
            log_file = f'logs/security/security_{datetime.now().strftime("%Y%m%d")}.log'
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # Rotar logs si es necesario
            self._rotate_logs()
        except Exception as e:
            logging.error(f"Error al registrar evento de seguridad: {str(e)}")
    
    def _rotate_logs(self) -> None:
        """Rota los logs de seguridad antiguos."""
        try:
            log_dir = 'logs/security'
            max_days = 30
            
            for file in os.listdir(log_dir):
                if file.startswith('security_') and file.endswith('.log'):
                    file_path = os.path.join(log_dir, file)
                    file_date = datetime.strptime(file.split('_')[1].split('.')[0], '%Y%m%d')
                    
                    if (datetime.now() - file_date).days > max_days:
                        os.remove(file_path)
                        logging.info(f"Log de seguridad eliminado: {file}")
        except Exception as e:
            logging.error(f"Error al rotar logs: {str(e)}")

class SecurityError(Exception):
    """Excepción personalizada para errores de seguridad."""
    pass

class ProductionModel:
    """
    Clase para el modelo en producción que incluye validaciones, monitoreo y predicciones.
    """
    
    def __init__(self, 
                 model_path: str = 'models/xgboost_model.pkl',
                 scaler_path: str = 'models/scaler.pkl',
                 threshold_rmse: float = 100.0,
                 threshold_mae: float = 50.0,
                 drift_threshold: float = 0.1):
        """
        Inicializa el modelo de producción.
        
        Args:
            model_path: Ruta al modelo entrenado
            scaler_path: Ruta al scaler entrenado
            threshold_rmse: Umbral de RMSE para alertas
            threshold_mae: Umbral de MAE para alertas
            drift_threshold: Umbral para detección de drift
        """
        try:
            logger.info("Inicializando modelo de producción")
            
            # Cargar modelo y scaler
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Cargar configuración de features
            self.feature_importance = pd.read_csv('results/feature_importance.csv')
            self.top_features = self.feature_importance.head(15)['Feature'].tolist()
            
            # Configurar umbrales
            self.threshold_rmse = threshold_rmse
            self.threshold_mae = threshold_mae
            self.drift_threshold = drift_threshold
            
            # Crear directorios necesarios
            self.results_dir = Path('results/production')
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Modelo de producción inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar el modelo: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Valida los datos de entrada.
        
        Args:
            df: DataFrame con los datos a validar
            
        Returns:
            Tuple[bool, str]: (es_válido, mensaje)
        """
        try:
            # Verificar columnas requeridas
            missing_cols = [col for col in self.top_features if col not in df.columns]
            if missing_cols:
                return False, f"Columnas faltantes: {missing_cols}"
            
            # Verificar valores negativos en columnas numéricas
            numeric_cols = df[self.top_features].select_dtypes(include=[np.number]).columns
            neg_counts = (df[numeric_cols] < 0).sum()
            if neg_counts.any():
                return False, f"Valores negativos encontrados en: {neg_counts[neg_counts > 0].to_dict()}"
            
            # Verificar tipos de datos
            for col in self.top_features:
                if col in numeric_cols and not np.issubdtype(df[col].dtype, np.number):
                    return False, f"Tipo de dato incorrecto en columna {col}"
            
            # Registrar columnas con valores nulos (pero no fallar)
            null_counts = df[self.top_features].isnull().sum()
            if null_counts.any():
                logger.warning(f"Se encontraron valores nulos que serán imputados en: {null_counts[null_counts > 0].to_dict()}")
            
            return True, "Datos válidos"
            
        except Exception as e:
            logger.error(f"Error en validación de datos: {str(e)}")
            return False, str(e)
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepara las features para la predicción.
        
        Args:
            df: DataFrame con los datos de entrada
            
        Returns:
            np.ndarray: Features preparadas
        """
        try:
            # Seleccionar features
            X = df[self.top_features].copy()
            
            # Imputar valores nulos con la mediana
            for col in X.columns:
                if X[col].isnull().any():
                    logger.info(f"Imputando valores nulos en columna {col} con la mediana")
                    X[col] = X[col].fillna(X[col].median())
            
            # Escalar features
            X_scaled = self.scaler.transform(X)
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Error al preparar features: {str(e)}")
            raise
    
    def generate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera predicciones para los datos de entrada.
        
        Args:
            df: DataFrame con los datos de entrada
            
        Returns:
            pd.DataFrame: DataFrame con predicciones
        """
        try:
            # Validar datos
            is_valid, message = self.validate_data(df)
            if not is_valid:
                raise ValueError(f"Datos inválidos: {message}")
            
            # Preparar features
            X_scaled = self.prepare_features(df)
            
            # Generar predicciones
            predictions = self.model.predict(X_scaled)
            
            # Crear DataFrame de resultados
            results = df.copy()
            results['predicted_importe_fx'] = predictions
            results['prediction_timestamp'] = datetime.now()
            
            return results
            
        except Exception as e:
            logger.error(f"Error al generar predicciones: {str(e)}")
            raise
    
    def detect_data_drift(self, df: pd.DataFrame) -> Dict:
        """
        Detecta drift en los datos de entrada.
        
        Args:
            df: DataFrame con los datos de entrada
            
        Returns:
            Dict: Resultados del análisis de drift
        """
        try:
            # Cargar datos de entrenamiento para comparación
            train_data = pd.read_csv('data/processed/features_data.csv', low_memory=False)
            
            drift_results = {}
            
            # Analizar drift en cada feature
            for feature in self.top_features:
                if feature in train_data.columns:
                    # Calcular estadísticas
                    train_mean = train_data[feature].mean()
                    train_std = train_data[feature].std()
                    current_mean = df[feature].mean()
                    current_std = df[feature].std()
                    
                    # Calcular drift
                    mean_drift = abs(current_mean - train_mean) / train_mean
                    std_drift = abs(current_std - train_std) / train_std
                    has_drift = mean_drift > self.drift_threshold or std_drift > self.drift_threshold
                    
                    drift_results[feature] = {
                        'mean_drift': float(mean_drift),
                        'std_drift': float(std_drift),
                        'has_drift': int(has_drift)  # Convertir bool a int
                    }
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Error al detectar drift: {str(e)}")
            raise
    
    def evaluate_predictions(self, df: pd.DataFrame) -> Dict:
        """
        Evalúa las predicciones generadas.
        
        Args:
            df: DataFrame con predicciones y valores reales
            
        Returns:
            Dict: Métricas de evaluación
        """
        try:
            # Calcular métricas
            y_true = df['Importe FX']
            y_pred = df['predicted_importe_fx']
            
            metrics = {
                'rmse': float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                'mae': float(np.mean(np.abs(y_true - y_pred))),
                'r2': float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)),
                'mape': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
            }
            
            # Verificar umbrales y convertir booleanos a enteros
            metrics['rmse_alert'] = int(metrics['rmse'] > self.threshold_rmse)
            metrics['mae_alert'] = int(metrics['mae'] > self.threshold_mae)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error al evaluar predicciones: {str(e)}")
            raise
    
    def save_predictions(self, df: pd.DataFrame, filename: Optional[str] = None):
        """
        Guarda las predicciones generadas.
        
        Args:
            df: DataFrame con predicciones
            filename: Nombre opcional del archivo
        """
        try:
            if filename is None:
                filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Guardar predicciones
            df.to_csv(self.results_dir / filename, index=False)
            logger.info(f"Predicciones guardadas en {filename}")
            
        except Exception as e:
            logger.error(f"Error al guardar predicciones: {str(e)}")
            raise
    
    def save_metrics(self, metrics: Dict, drift_results: Dict):
        """
        Guarda las métricas de evaluación y drift.
        
        Args:
            metrics: Diccionario con métricas de evaluación
            drift_results: Diccionario con resultados de drift
        """
        try:
            # Convertir booleanos a enteros para serialización JSON
            metrics_json = {}
            for key, value in metrics.items():
                if isinstance(value, bool):
                    metrics_json[key] = int(value)
                else:
                    metrics_json[key] = value
            
            drift_results_json = {}
            for feature, results in drift_results.items():
                drift_results_json[feature] = {
                    k: int(v) if isinstance(v, bool) else v
                    for k, v in results.items()
                }
            
            # Crear diccionario con resultados
            results = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics_json,
                'drift_results': drift_results_json
            }
            
            # Guardar resultados
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(self.results_dir / filename, 'w') as f:
                json.dump(results, f, indent=4)
            
            logger.info(f"Métricas guardadas en {filename}")
            
        except Exception as e:
            logger.error(f"Error al guardar métricas: {str(e)}")
            raise
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Obtiene la importancia de las features.
        
        Returns:
            pd.DataFrame: DataFrame con importancia de features
        """
        return self.feature_importance
    
    def generate_dashboard(self, df: pd.DataFrame, metrics: Dict, drift_results: Dict):
        """
        Genera un dashboard con visualizaciones de las predicciones.
        
        Args:
            df: DataFrame con predicciones
            metrics: Diccionario con métricas
            drift_results: Diccionario con resultados de drift
        """
        try:
            # Crear directorio para gráficos
            plots_dir = self.results_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # 1. Gráfico de predicciones vs valores reales
            plt.figure(figsize=(10, 6))
            plt.scatter(df['Importe FX'], df['predicted_importe_fx'], alpha=0.5)
            plt.plot([df['Importe FX'].min(), df['Importe FX'].max()], 
                    [df['Importe FX'].min(), df['Importe FX'].max()], 
                    'r--', label='Línea perfecta')
            plt.xlabel('Valores Reales')
            plt.ylabel('Predicciones')
            plt.title('Predicciones vs Valores Reales')
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / 'predictions_vs_real.png')
            plt.close()
            
            # 2. Gráfico de errores
            plt.figure(figsize=(10, 6))
            errors = df['Importe FX'] - df['predicted_importe_fx']
            sns.histplot(errors, bins=50)
            plt.title('Distribución de Errores')
            plt.tight_layout()
            plt.savefig(plots_dir / 'error_distribution.png')
            plt.close()
            
            # 3. Gráfico de drift
            drift_df = pd.DataFrame([
                {'feature': feature, 'mean_drift': results['mean_drift']}
                for feature, results in drift_results.items()
            ])
            plt.figure(figsize=(12, 6))
            sns.barplot(data=drift_df, x='feature', y='mean_drift')
            plt.xticks(rotation=45)
            plt.title('Drift por Feature')
            plt.tight_layout()
            plt.savefig(plots_dir / 'feature_drift.png')
            plt.close()
            
            logger.info("Dashboard generado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al generar dashboard: {str(e)}")
            raise

def main():
    """
    Función principal para ejecutar el modelo en producción.
    """
    try:
        # Inicializar modelo
        model = ProductionModel()
        
        # Cargar datos de prueba
        df = pd.read_csv('data/processed/features_data.csv', low_memory=False)
        
        # Generar predicciones
        results = model.generate_predictions(df)
        
        # Detectar drift
        drift_results = model.detect_data_drift(df)
        
        # Evaluar predicciones
        metrics = model.evaluate_predictions(results)
        
        # Guardar resultados
        model.save_predictions(results)
        model.save_metrics(metrics, drift_results)
        
        # Generar dashboard
        model.generate_dashboard(results, metrics, drift_results)
        
        logger.info("Ejecución del modelo en producción completada exitosamente")
        
    except Exception as e:
        logger.error(f"Error en ejecución principal: {str(e)}")
        raise

if __name__ == "__main__":
    main() 