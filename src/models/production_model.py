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

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production.log'),
        logging.StreamHandler()
    ]
)

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
    def __init__(self, model_path: str, scaler_path: Optional[str] = None, 
                 threshold_rmse: float = 1000, threshold_mae: float = 500,
                 drift_threshold: float = 0.1):
        """
        Inicializa el modelo de producción.
        
        Args:
            model_path: Ruta al modelo guardado
            scaler_path: Ruta al scaler guardado (opcional)
            threshold_rmse: Umbral de alerta para RMSE
            threshold_mae: Umbral de alerta para MAE
            drift_threshold: Umbral para detección de drift
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.threshold_rmse = threshold_rmse
        self.threshold_mae = threshold_mae
        self.drift_threshold = drift_threshold
        self.security_manager = SecurityManager()
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.predictions_history = []
        self.metrics_history = []
        self.feature_stats_history = []
        
    def _load_model(self) -> object:
        """
        Carga el modelo de forma segura.
        
        Returns:
            object: Modelo cargado
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado en {self.model_path}")
            
            # Verificar integridad del modelo
            model_hash = self.security_manager.generate_hash(open(self.model_path, 'rb').read())
            if not self._verify_model_integrity(model_hash):
                raise SecurityError("Integridad del modelo comprometida")
            
            return joblib.load(self.model_path)
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {str(e)}")
            raise
    
    def _load_scaler(self) -> Optional[object]:
        """
        Carga el scaler de forma segura.
        
        Returns:
            Optional[object]: Scaler cargado o None
        """
        try:
            if not self.scaler_path or not os.path.exists(self.scaler_path):
                return None
            
            # Verificar integridad del scaler
            scaler_hash = self.security_manager.generate_hash(open(self.scaler_path, 'rb').read())
            if not self._verify_scaler_integrity(scaler_hash):
                raise SecurityError("Integridad del scaler comprometida")
            
            return joblib.load(self.scaler_path)
        except Exception as e:
            logging.error(f"Error al cargar el scaler: {str(e)}")
            raise
    
    def _verify_model_integrity(self, current_hash: str) -> bool:
        """
        Verifica la integridad del modelo.
        
        Args:
            current_hash: Hash actual del modelo
            
        Returns:
            bool: True si la integridad está verificada
        """
        try:
            hash_file = f"{self.model_path}.hash"
            if not os.path.exists(hash_file):
                with open(hash_file, 'w') as f:
                    f.write(current_hash)
                return True
            
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            
            return current_hash == stored_hash
        except Exception as e:
            logging.error(f"Error al verificar integridad del modelo: {str(e)}")
            return False
    
    def _verify_scaler_integrity(self, current_hash: str) -> bool:
        """
        Verifica la integridad del scaler.
        
        Args:
            current_hash: Hash actual del scaler
            
        Returns:
            bool: True si la integridad está verificada
        """
        try:
            hash_file = f"{self.scaler_path}.hash"
            if not os.path.exists(hash_file):
                with open(hash_file, 'w') as f:
                    f.write(current_hash)
                return True
            
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            
            return current_hash == stored_hash
        except Exception as e:
            logging.error(f"Error al verificar integridad del scaler: {str(e)}")
            return False
    
    def detect_data_drift(self, df: pd.DataFrame) -> Dict:
        """
        Detecta drift en los datos de entrada.
        
        Args:
            df: DataFrame con los datos actuales
            
        Returns:
            Dict con métricas de drift
        """
        try:
            drift_metrics = {}
            
            # Calcular estadísticas actuales
            current_stats = df.describe()
            
            # Cargar estadísticas históricas
            stats_path = Path('results/feature_stats_history.json')
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    historical_stats = json.load(f)
                
                # Comparar con estadísticas históricas
                for column in current_stats.columns:
                    if column in historical_stats[-1]:
                        mean_diff = abs(current_stats[column]['mean'] - historical_stats[-1][column]['mean'])
                        std_diff = abs(current_stats[column]['std'] - historical_stats[-1][column]['std'])
                        
                        drift_metrics[column] = {
                            'mean_drift': mean_diff,
                            'std_drift': std_diff,
                            'drift_detected': mean_diff > self.drift_threshold or std_diff > self.drift_threshold
                        }
            
            # Guardar estadísticas actuales
            self.feature_stats_history.append(current_stats.to_dict())
            self.save_feature_stats()
            
            return drift_metrics
        except Exception as e:
            logging.error(f"Error al detectar drift: {str(e)}")
            raise
    
    def save_feature_stats(self) -> None:
        """
        Guarda el historial de estadísticas de features.
        """
        try:
            stats_path = Path('results/feature_stats_history.json')
            with open(stats_path, 'w') as f:
                json.dump(self.feature_stats_history, f, indent=4)
            
            logging.info(f"Estadísticas de features guardadas en {stats_path}")
        except Exception as e:
            logging.error(f"Error al guardar estadísticas de features: {str(e)}")
            raise
    
    def generate_dashboard(self) -> None:
        """
        Genera un dashboard interactivo con las métricas y predicciones.
        """
        try:
            # Crear figura con subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Predicciones vs Valores Reales', 
                              'Métricas de Rendimiento',
                              'Importancia de Features',
                              'Drift de Datos')
            )
            
            # Optimizar datos para visualización
            if self.predictions_history:
                # Limitar a últimos 30 días para mejor rendimiento
                pred_df = pd.DataFrame(self.predictions_history[-30:])
                pred_df['Fecha'] = pd.to_datetime(pred_df['Fecha'])
                
                # Agregar predicciones
                fig.add_trace(
                    go.Scatter(x=pred_df['Fecha'], y=pred_df['Predicción'],
                              name='Predicción', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # Agregar intervalos de confianza
                fig.add_trace(
                    go.Scatter(x=pred_df['Fecha'], y=pred_df['Intervalo_Superior'],
                              fill=None, mode='lines', line_color='rgba(0,100,80,0.2)',
                              name='Intervalo Superior'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=pred_df['Fecha'], y=pred_df['Intervalo_Inferior'],
                              fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)',
                              name='Intervalo Inferior'),
                    row=1, col=1
                )
            
            # Gráfico de métricas optimizado
            if self.metrics_history:
                metrics_df = pd.DataFrame(self.metrics_history[-30:])
                metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
                
                # Agregar métricas con colores diferentes
                fig.add_trace(
                    go.Scatter(x=metrics_df['timestamp'], y=metrics_df['rmse'],
                              name='RMSE', line=dict(color='red')),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=metrics_df['timestamp'], y=metrics_df['mae'],
                              name='MAE', line=dict(color='green')),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=metrics_df['timestamp'], y=metrics_df['r2'],
                              name='R²', line=dict(color='blue')),
                    row=2, col=1
                )
            
            # Gráfico de importancia de features optimizado
            importance = self.get_feature_importance()
            fig.add_trace(
                go.Bar(x=importance['feature'], y=importance['importance'],
                      name='Importancia', marker_color='rgb(55, 83, 109)'),
                row=1, col=2
            )
            
            # Gráfico de drift optimizado
            if self.feature_stats_history:
                drift_df = pd.DataFrame([
                    {col: stats[col]['mean'] 
                     for col in stats.keys()}
                    for stats in self.feature_stats_history[-30:]
                ])
                fig.add_trace(
                    go.Heatmap(z=drift_df.values,
                              x=drift_df.columns,
                              y=drift_df.index,
                              name='Drift',
                              colorscale='RdBu'),
                    row=2, col=2
                )
            
            # Actualizar layout con mejoras visuales
            fig.update_layout(
                height=800,
                title_text="Dashboard de Monitoreo del Modelo",
                showlegend=True,
                template='plotly_white',
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            # Actualizar ejes y títulos
            fig.update_xaxes(title_text="Fecha", row=1, col=1)
            fig.update_xaxes(title_text="Fecha", row=2, col=1)
            fig.update_xaxes(title_text="Features", row=1, col=2)
            fig.update_xaxes(title_text="Features", row=2, col=2)
            
            fig.update_yaxes(title_text="Volumen", row=1, col=1)
            fig.update_yaxes(title_text="Métricas", row=2, col=1)
            fig.update_yaxes(title_text="Importancia", row=1, col=2)
            fig.update_yaxes(title_text="Días", row=2, col=2)
            
            # Guardar dashboard con configuración optimizada
            dashboard_path = Path('results/dashboard.html')
            fig.write_html(
                dashboard_path,
                config={'responsive': True},
                include_plotlyjs='cdn',
                full_html=True
            )
            
            logging.info(f"Dashboard generado en {dashboard_path}")
        except Exception as e:
            logging.error(f"Error al generar dashboard: {str(e)}")
            raise
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimiza el DataFrame para mejor rendimiento.
        
        Args:
            df: DataFrame a optimizar
            
        Returns:
            DataFrame optimizado
        """
        try:
            # Optimizar tipos de datos
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
                elif df[col].dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                elif df[col].dtype == 'object':
                    if df[col].nunique() / len(df) < 0.5:
                        df[col] = df[col].astype('category')
            
            return df
        except Exception as e:
            logging.error(f"Error al optimizar DataFrame: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Valida los datos de entrada.
        
        Args:
            df: DataFrame con los datos
            
        Returns:
            Tuple con (bool indicando si los datos son válidos, lista de errores)
        """
        try:
            errors = []
            
            # Verificar columnas requeridas
            required_columns = ['Importe FX', 'Volumen_promedio_diario', 'Spread_TC', 
                              'Volumen_Ponderado_5', 'Volumen_diario']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                errors.append(f"Columnas faltantes: {missing_columns}")
            
            # Verificar valores nulos
            null_counts = df[required_columns].isnull().sum()
            if null_counts.any():
                errors.append(f"Valores nulos encontrados: {null_counts[null_counts > 0].to_dict()}")
            
            # Verificar valores negativos
            for col in required_columns:
                if (df[col] < 0).any():
                    errors.append(f"Valores negativos encontrados en columna {col}")
            
            # Verificar outliers
            for col in required_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                if not outliers.empty:
                    errors.append(f"Outliers detectados en columna {col}: {len(outliers)} registros")
            
            # Verificar tipos de datos
            for col in required_columns:
                if not np.issubdtype(df[col].dtype, np.number):
                    errors.append(f"Tipo de dato incorrecto en columna {col}")
            
            # Verificar rangos de valores
            if (df['Spread_TC'] > 1).any():
                errors.append("Valores de Spread_TC fuera de rango (> 1)")
            
            return len(errors) == 0, errors
        except Exception as e:
            logging.error(f"Error en validación de datos: {str(e)}")
            raise
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara las features para el modelo.
        
        Args:
            df: DataFrame con los datos
            
        Returns:
            DataFrame con features preparadas
        """
        try:
            # Validar datos
            is_valid, errors = self.validate_data(df)
            if not is_valid:
                raise ValueError(f"Error de validación: {errors}")
            
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
            
            # Normalizar features si hay scaler
            if self.scaler is not None:
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                df[numeric_columns] = self.scaler.transform(df[numeric_columns])
            
            return df
        except Exception as e:
            logging.error(f"Error al preparar features: {str(e)}")
            raise
    
    def generate_predictions(self, df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
        """
        Genera predicciones para los próximos días.
        
        Args:
            df: DataFrame con datos históricos
            horizon: Número de días a predecir
            
        Returns:
            DataFrame con predicciones
        """
        try:
            # Optimizar DataFrame
            df = self._optimize_dataframe(df)
            
            # Preparar features
            X = self.prepare_features(df)
            
            # Generar predicciones
            predictions = self.model.predict(X)
            
            # Calcular intervalos de confianza dinámicos
            std_dev = np.std(predictions)
            confidence_intervals = 1.96 * std_dev / np.sqrt(len(predictions))
            
            # Crear DataFrame con predicciones
            results = pd.DataFrame({
                'Fecha': df['Fecha de cierre'],
                'Predicción': predictions,
                'Intervalo_Inferior': predictions - confidence_intervals,
                'Intervalo_Superior': predictions + confidence_intervals,
                'Confianza': 0.95
            })
            
            # Optimizar DataFrame de resultados
            results = self._optimize_dataframe(results)
            
            # Guardar predicciones
            self.save_predictions(results)
            
            return results
        except Exception as e:
            logging.error(f"Error al generar predicciones: {str(e)}")
            raise
    
    def evaluate_predictions(self, predictions: pd.DataFrame, actual_values: pd.Series) -> Dict:
        """
        Evalúa las predicciones contra valores reales.
        
        Args:
            predictions: DataFrame con predicciones
            actual_values: Serie con valores reales
            
        Returns:
            Dict con métricas de evaluación
        """
        try:
            metrics = {
                'rmse': np.sqrt(mean_squared_error(actual_values, predictions['Predicción'])),
                'mae': mean_absolute_error(actual_values, predictions['Predicción']),
                'r2': r2_score(actual_values, predictions['Predicción']),
                'timestamp': datetime.now().isoformat()
            }
            
            # Guardar métricas
            self.save_metrics(metrics)
            
            return metrics
        except Exception as e:
            logging.error(f"Error al evaluar predicciones: {str(e)}")
            raise
    
    def save_predictions(self, predictions: pd.DataFrame) -> None:
        """
        Guarda las predicciones en el historial.
        
        Args:
            predictions: DataFrame con predicciones
        """
        try:
            # Convertir predicciones a dict
            pred_dict = predictions.to_dict(orient='records')
            
            # Agregar timestamp
            for pred in pred_dict:
                pred['timestamp'] = datetime.now().isoformat()
            
            # Agregar al historial
            self.predictions_history.extend(pred_dict)
            
            # Guardar en archivo
            predictions_path = Path('results/predictions_history.json')
            with open(predictions_path, 'w') as f:
                json.dump(self.predictions_history, f, indent=4)
            
            logging.info(f"Predicciones guardadas en {predictions_path}")
        except Exception as e:
            logging.error(f"Error al guardar predicciones: {str(e)}")
            raise
    
    def save_metrics(self, metrics: Dict) -> None:
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
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Obtiene la importancia de las features.
        
        Returns:
            DataFrame con importancia de features
        """
        try:
            importance = pd.DataFrame({
                'feature': self.model.feature_names_in_,
                'importance': self.model.feature_importances_
            })
            importance = importance.sort_values('importance', ascending=False)
            
            # Guardar importancia
            importance_path = Path('results/feature_importance_production.csv')
            importance.to_csv(importance_path, index=False)
            
            return importance
        except Exception as e:
            logging.error(f"Error al obtener importancia de features: {str(e)}")
            raise

def main():
    """
    Función principal que ejecuta el modelo en producción.
    """
    try:
        # Crear directorios necesarios
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Inicializar modelo de producción
        model = ProductionModel(
            model_path='models/xgboost_model.joblib',
            scaler_path='models/scaler.joblib'
        )
        
        # Cargar datos recientes
        data_path = os.path.join('data', 'processed', 'features.csv')
        df = pd.read_csv(data_path)
        df['Fecha de cierre'] = pd.to_datetime(df['Fecha de cierre'])
        
        # Detectar drift
        drift_metrics = model.detect_data_drift(df)
        if any(metric['drift_detected'] for metric in drift_metrics.values()):
            logging.warning(f"Drift detectado en features: {drift_metrics}")
        
        # Generar predicciones
        predictions = model.generate_predictions(df)
        
        # Obtener importancia de features
        importance = model.get_feature_importance()
        
        # Generar dashboard
        model.generate_dashboard()
        
        # Logging de resultados
        logging.info("\nPredicciones generadas:")
        logging.info(f"Número de predicciones: {len(predictions)}")
        logging.info("\nTop 5 features más importantes:")
        logging.info(importance.head().to_string())
        
    except Exception as e:
        logging.error(f"Error en el proceso principal: {str(e)}")
        raise

if __name__ == "__main__":
    main() 