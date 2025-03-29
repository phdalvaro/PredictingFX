import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import joblib
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluate_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_data():
    """
    Carga el modelo entrenado y los datos de prueba.
    """
    try:
        logger.info("Cargando modelo y datos")
        
        # Cargar modelo y scaler
        model = joblib.load('models/xgboost_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Cargar datos de prueba
        df = pd.read_csv('data/processed/features_data.csv', low_memory=False)
        feature_importance = pd.read_csv('results/feature_importance.csv')
        top_features = feature_importance.head(15)['Feature'].tolist()
        
        # Preparar datos
        X = df[top_features]
        y = df['Importe FX']
        X_scaled = scaler.transform(X)
        
        return model, X_scaled, y, top_features
        
    except Exception as e:
        logger.error(f"Error al cargar modelo y datos: {str(e)}")
        raise

def calculate_error_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calcula métricas de error detalladas.
    """
    try:
        # Métricas básicas
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Error porcentual absoluto medio (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Error máximo
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Error mediano
        median_error = np.median(np.abs(y_true - y_pred))
        
        # Desviación estándar del error
        error_std = np.std(y_true - y_pred)
        
        # Intervalos de confianza para el error
        error_ci = stats.t.interval(0.95, len(y_true)-1, loc=np.mean(y_true - y_pred), scale=stats.sem(y_true - y_pred))
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Max_Error': max_error,
            'Median_Error': median_error,
            'Error_Std': error_std,
            'Error_CI_Lower': error_ci[0],
            'Error_CI_Upper': error_ci[1]
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error al calcular métricas: {str(e)}")
        raise

def analyze_error_distribution(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Analiza la distribución de errores.
    """
    try:
        errors = y_true - y_pred
        
        # Estadísticas de la distribución
        stats_dict = {
            'Mean': np.mean(errors),
            'Std': np.std(errors),
            'Skewness': stats.skew(errors),
            'Kurtosis': stats.kurtosis(errors),
            'Shapiro_Wilk_pvalue': stats.shapiro(errors)[1]
        }
        
        # Prueba de normalidad
        is_normal = stats_dict['Shapiro_Wilk_pvalue'] > 0.05
        
        return stats_dict, is_normal
        
    except Exception as e:
        logger.error(f"Error al analizar distribución de errores: {str(e)}")
        raise

def analyze_bias(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Analiza el sesgo en las predicciones.
    """
    try:
        # Calcular sesgo por rangos de valores
        ranges = np.percentile(y_true, [0, 25, 50, 75, 100])
        biases = []
        
        for i in range(len(ranges)-1):
            mask = (y_true >= ranges[i]) & (y_true < ranges[i+1])
            if np.sum(mask) > 0:
                bias = np.mean(y_pred[mask] - y_true[mask])
                biases.append({
                    'Range': f'{ranges[i]:.2f}-{ranges[i+1]:.2f}',
                    'Bias': bias,
                    'Count': np.sum(mask)
                })
        
        return pd.DataFrame(biases)
        
    except Exception as e:
        logger.error(f"Error al analizar sesgo: {str(e)}")
        raise

def plot_evaluation_results(y_true: np.ndarray, y_pred: np.ndarray, error_stats: dict, 
                          bias_analysis: pd.DataFrame):
    """
    Genera gráficos de evaluación.
    """
    try:
        logger.info("Generando gráficos de evaluación")
        
        # Crear directorio para gráficos
        plots_dir = Path('results/evaluation_plots')
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Gráfico de errores vs valores reales
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_true - y_pred, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Valores Reales')
        plt.ylabel('Errores')
        plt.title('Errores vs Valores Reales')
        plt.tight_layout()
        plt.savefig(plots_dir / 'errors_vs_real.png')
        plt.close()
        
        # 2. Histograma de errores
        plt.figure(figsize=(10, 6))
        sns.histplot(y_true - y_pred, bins=50)
        plt.title('Distribución de Errores')
        plt.tight_layout()
        plt.savefig(plots_dir / 'error_distribution.png')
        plt.close()
        
        # 3. Gráfico de sesgo por rangos
        plt.figure(figsize=(10, 6))
        sns.barplot(data=bias_analysis, x='Range', y='Bias')
        plt.xticks(rotation=45)
        plt.title('Sesgo por Rangos de Valores')
        plt.tight_layout()
        plt.savefig(plots_dir / 'bias_by_range.png')
        plt.close()
        
        # 4. Gráfico de Q-Q
        plt.figure(figsize=(10, 6))
        stats.probplot(y_true - y_pred, dist="norm", plot=plt)
        plt.title('Gráfico Q-Q de Errores')
        plt.tight_layout()
        plt.savefig(plots_dir / 'qq_plot.png')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error al generar gráficos: {str(e)}")
        raise

def save_evaluation_results(metrics: dict, error_stats: dict, bias_analysis: pd.DataFrame):
    """
    Guarda los resultados de la evaluación.
    """
    try:
        # Crear directorio para resultados
        results_dir = Path('results/evaluation')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar métricas
        pd.DataFrame([metrics]).to_csv(results_dir / 'metrics.csv', index=False)
        
        # Guardar estadísticas de error
        pd.DataFrame([error_stats]).to_csv(results_dir / 'error_stats.csv', index=False)
        
        # Guardar análisis de sesgo
        bias_analysis.to_csv(results_dir / 'bias_analysis.csv', index=False)
        
        logger.info("Resultados guardados exitosamente")
        
    except Exception as e:
        logger.error(f"Error al guardar resultados: {str(e)}")
        raise

def main():
    """
    Función principal para ejecutar la evaluación del modelo.
    """
    try:
        # Cargar modelo y datos
        model, X_scaled, y, feature_names = load_model_and_data()
        
        # Realizar predicciones
        y_pred = model.predict(X_scaled)
        
        # Calcular métricas
        metrics = calculate_error_metrics(y, y_pred)
        logger.info("\nMétricas de rendimiento:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Analizar distribución de errores
        error_stats, is_normal = analyze_error_distribution(y, y_pred)
        logger.info("\nEstadísticas de error:")
        for stat, value in error_stats.items():
            logger.info(f"{stat}: {value:.4f}")
        logger.info(f"Distribución normal: {is_normal}")
        
        # Analizar sesgo
        bias_analysis = analyze_bias(y, y_pred)
        logger.info("\nAnálisis de sesgo por rangos:")
        logger.info(bias_analysis.to_string())
        
        # Generar gráficos
        plot_evaluation_results(y, y_pred, error_stats, bias_analysis)
        
        # Guardar resultados
        save_evaluation_results(metrics, error_stats, bias_analysis)
        
        logger.info("Evaluación del modelo completada exitosamente")
        
    except Exception as e:
        logger.error(f"Error en ejecución principal: {str(e)}")
        raise

if __name__ == "__main__":
    main() 