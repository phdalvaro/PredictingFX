import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitor_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """
    Clase para monitorear el rendimiento del modelo en producción.
    """
    
    def __init__(self,
                 results_dir: str = 'results/production',
                 alert_thresholds: Optional[Dict] = None,
                 email_config: Optional[Dict] = None):
        """
        Inicializa el monitor del modelo.
        
        Args:
            results_dir: Directorio con los resultados de producción
            alert_thresholds: Umbrales para alertas
            email_config: Configuración para envío de alertas por email
        """
        self.results_dir = Path(results_dir)
        self.alert_thresholds = alert_thresholds or {
            'rmse': 100.0,
            'mae': 50.0,
            'mape': 5.0,
            'drift': 0.1
        }
        self.email_config = email_config
        self._setup_monitoring()
    
    def _setup_monitoring(self) -> None:
        """Configura el sistema de monitoreo."""
        try:
            # Crear directorios necesarios
            self.results_dir.mkdir(parents=True, exist_ok=True)
            self.monitoring_dir = self.results_dir / 'monitoring'
            self.monitoring_dir.mkdir(exist_ok=True)
            
            # Crear archivo de historial si no existe
            self.history_file = self.monitoring_dir / 'monitoring_history.json'
            if not self.history_file.exists():
                self._save_history([])
            
            logger.info("Sistema de monitoreo configurado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al configurar monitoreo: {str(e)}")
            raise
    
    def _load_history(self) -> List[Dict]:
        """Carga el historial de monitoreo."""
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error al cargar historial: {str(e)}")
            return []
    
    def _save_history(self, history: List[Dict]) -> None:
        """Guarda el historial de monitoreo."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=4)
        except Exception as e:
            logger.error(f"Error al guardar historial: {str(e)}")
            raise
    
    def _get_latest_metrics(self) -> Optional[Dict]:
        """Obtiene las métricas más recientes."""
        try:
            # Obtener archivos de métricas
            metric_files = list(self.results_dir.glob('metrics_*.json'))
            if not metric_files:
                return None
            
            # Obtener el archivo más reciente
            latest_file = max(metric_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                return json.load(f)
            
        except Exception as e:
            logger.error(f"Error al obtener métricas recientes: {str(e)}")
            return None
    
    def _check_alerts(self, metrics: Dict) -> List[str]:
        """
        Verifica si hay alertas basadas en las métricas.
        
        Args:
            metrics: Diccionario con métricas
            
        Returns:
            List[str]: Lista de mensajes de alerta
        """
        alerts = []
        
        try:
            # Verificar RMSE
            if metrics['metrics']['rmse'] > self.alert_thresholds['rmse']:
                alerts.append(f"RMSE ({metrics['metrics']['rmse']:.2f}) excede el umbral ({self.alert_thresholds['rmse']})")
            
            # Verificar MAE
            if metrics['metrics']['mae'] > self.alert_thresholds['mae']:
                alerts.append(f"MAE ({metrics['metrics']['mae']:.2f}) excede el umbral ({self.alert_thresholds['mae']})")
            
            # Verificar MAPE
            if metrics['metrics']['mape'] > self.alert_thresholds['mape']:
                alerts.append(f"MAPE ({metrics['metrics']['mape']:.2f}%) excede el umbral ({self.alert_thresholds['mape']}%)")
            
            # Verificar drift
            for feature, drift in metrics['drift_results'].items():
                if drift['mean_drift'] > self.alert_thresholds['drift']:
                    alerts.append(f"Drift detectado en feature {feature}: {drift['mean_drift']:.2f}")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error al verificar alertas: {str(e)}")
            return []
    
    def _send_alert_email(self, alerts: List[str], metrics: Dict) -> None:
        """
        Envía alertas por email.
        
        Args:
            alerts: Lista de mensajes de alerta
            metrics: Diccionario con métricas
        """
        if not self.email_config:
            return
        
        try:
            # Crear mensaje
            msg = MIMEMultipart()
            msg['Subject'] = 'Alertas de Monitoreo del Modelo FX'
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            
            # Crear cuerpo del mensaje
            body = "Se han detectado las siguientes alertas:\n\n"
            for alert in alerts:
                body += f"- {alert}\n"
            
            body += "\nMétricas actuales:\n"
            for metric, value in metrics['metrics'].items():
                body += f"- {metric}: {value:.4f}\n"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Enviar email
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
            
            logger.info("Alertas enviadas por email exitosamente")
            
        except Exception as e:
            logger.error(f"Error al enviar alertas por email: {str(e)}")
    
    def _generate_monitoring_plots(self, history: List[Dict]) -> None:
        """
        Genera gráficos de monitoreo.
        
        Args:
            history: Historial de monitoreo
        """
        try:
            # Crear DataFrame con historial
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 1. Gráfico de métricas a lo largo del tiempo
            plt.figure(figsize=(12, 6))
            metrics_data = []
            for _, row in df.iterrows():
                for metric, value in row['metrics'].items():
                    metrics_data.append({
                        'timestamp': row['timestamp'],
                        'metric': metric,
                        'value': value
                    })
            
            metrics_df = pd.DataFrame(metrics_data)
            sns.lineplot(data=metrics_df, x='timestamp', y='value', hue='metric')
            plt.title('Métricas de Rendimiento a lo largo del Tiempo')
            plt.xlabel('Fecha')
            plt.ylabel('Valor')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.monitoring_dir / 'metrics_over_time.png')
            plt.close()
            
            # 2. Gráfico de drift por feature
            drift_data = []
            for _, row in df.iterrows():
                for feature, drift in row['drift_results'].items():
                    drift_data.append({
                        'timestamp': row['timestamp'],
                        'feature': feature,
                        'mean_drift': drift['mean_drift']
                    })
            
            drift_df = pd.DataFrame(drift_data)
            drift_df['timestamp'] = pd.to_datetime(drift_df['timestamp'])
            
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=drift_df, x='timestamp', y='mean_drift', hue='feature')
            plt.title('Drift por Feature a lo largo del Tiempo')
            plt.xlabel('Fecha')
            plt.ylabel('Drift')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.monitoring_dir / 'drift_over_time.png')
            plt.close()
            
            logger.info("Gráficos de monitoreo generados exitosamente")
            
        except Exception as e:
            logger.error(f"Error al generar gráficos de monitoreo: {str(e)}")
    
    def run_monitoring(self) -> None:
        """Ejecuta el proceso de monitoreo."""
        try:
            logger.info("Iniciando monitoreo del modelo")
            
            # Obtener métricas recientes
            metrics = self._get_latest_metrics()
            if not metrics:
                logger.warning("No se encontraron métricas recientes")
                return
            
            # Verificar alertas
            alerts = self._check_alerts(metrics)
            
            # Si hay alertas, enviarlas por email
            if alerts:
                logger.warning(f"Alertas detectadas: {alerts}")
                self._send_alert_email(alerts, metrics)
            
            # Actualizar historial
            history = self._load_history()
            history.append(metrics)
            
            # Mantener solo los últimos 30 días
            cutoff_date = datetime.now() - timedelta(days=30)
            history = [
                entry for entry in history
                if datetime.fromisoformat(entry['timestamp']) > cutoff_date
            ]
            
            self._save_history(history)
            
            # Generar gráficos
            self._generate_monitoring_plots(history)
            
            logger.info("Monitoreo completado exitosamente")
            
        except Exception as e:
            logger.error(f"Error en monitoreo: {str(e)}")
            raise

def main():
    """
    Función principal para ejecutar el monitoreo.
    """
    try:
        # Configuración de email (opcional)
        email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('EMAIL_USERNAME'),
            'password': os.getenv('EMAIL_PASSWORD'),
            'from': os.getenv('EMAIL_FROM'),
            'to': os.getenv('EMAIL_TO')
        }
        
        # Inicializar monitor
        monitor = ModelMonitor(email_config=email_config)
        
        # Ejecutar monitoreo
        monitor.run_monitoring()
        
    except Exception as e:
        logger.error(f"Error en ejecución principal: {str(e)}")
        raise

if __name__ == "__main__":
    main() 