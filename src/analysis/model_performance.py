import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import statsmodels.api as sm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_performance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelPerformanceAnalyzer:
    """
    Clase para analizar el rendimiento histórico del modelo.
    """
    
    def __init__(self, results_dir: str = 'results/production'):
        """
        Inicializa el analizador de rendimiento.
        
        Args:
            results_dir: Directorio con los resultados de producción
        """
        self.results_dir = Path(results_dir)
        self.analysis_dir = self.results_dir / 'analysis'
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
    
    def load_predictions(self) -> pd.DataFrame:
        """
        Carga todas las predicciones históricas.
        
        Returns:
            pd.DataFrame: DataFrame con predicciones
        """
        try:
            # Cargar todos los archivos de predicciones
            prediction_files = list(self.results_dir.glob('predictions_*.csv'))
            if not prediction_files:
                raise FileNotFoundError("No se encontraron archivos de predicciones")
            
            # Combinar todos los archivos
            dfs = []
            for file in prediction_files:
                # Usar low_memory=False para evitar advertencias de tipos mixtos
                df = pd.read_csv(file, low_memory=False)
                df['prediction_file'] = file.name
                dfs.append(df)
            
            predictions_df = pd.concat(dfs, ignore_index=True)
            predictions_df['prediction_timestamp'] = pd.to_datetime(predictions_df['prediction_timestamp'])
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error al cargar predicciones: {str(e)}")
            raise
    
    def analyze_error_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analiza la distribución de errores.
        
        Args:
            df: DataFrame con predicciones
            
        Returns:
            Dict: Estadísticas de la distribución de errores
        """
        try:
            errors = df['Importe FX'] - df['predicted_importe_fx']
            
            # Estadísticas básicas
            stats = {
                'mean_error': float(errors.mean()),
                'std_error': float(errors.std()),
                'median_error': float(errors.median()),
                'skewness': float(errors.skew()),
                'kurtosis': float(errors.kurtosis()),
                'q1_error': float(errors.quantile(0.25)),
                'q3_error': float(errors.quantile(0.75))
            }
            
            # Test de normalidad
            _, p_value = sm.stats.diagnostic.lilliefors(errors)
            stats['normal_test_pvalue'] = float(p_value)
            
            # Gráfico de distribución de errores
            plt.figure(figsize=(10, 6))
            sns.histplot(errors, bins=50, kde=True)
            plt.title('Distribución de Errores de Predicción')
            plt.xlabel('Error')
            plt.ylabel('Frecuencia')
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'error_distribution.png')
            plt.close()
            
            # Q-Q plot
            plt.figure(figsize=(10, 6))
            sm.graphics.qqplot(errors, line='45')
            plt.title('Q-Q Plot de Errores')
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'error_qq_plot.png')
            plt.close()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error al analizar distribución de errores: {str(e)}")
            raise
    
    def analyze_error_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analiza patrones en los errores.
        
        Args:
            df: DataFrame con predicciones
            
        Returns:
            Dict: Resultados del análisis de patrones
        """
        try:
            df = df.copy()
            df['error'] = df['Importe FX'] - df['predicted_importe_fx']
            df['abs_error'] = abs(df['error'])
            df['error_pct'] = (df['error'] / df['Importe FX']) * 100
            
            # Análisis por rangos de valores
            value_ranges = pd.qcut(df['Importe FX'], q=5)
            error_by_range = df.groupby(value_ranges)['abs_error'].agg(['mean', 'std', 'count'])
            
            # Gráfico de errores por rango
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df, x=value_ranges, y='error')
            plt.title('Distribución de Errores por Rango de Valores')
            plt.xlabel('Rango de Valores')
            plt.ylabel('Error')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'errors_by_range.png')
            plt.close()
            
            # Análisis de autocorrelación de errores
            acf = sm.tsa.acf(df['error'], nlags=20)
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(acf)), acf)
            plt.title('Autocorrelación de Errores')
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelación')
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'error_autocorrelation.png')
            plt.close()
            
            return {
                'error_by_range': error_by_range.to_dict(),
                'autocorrelation': list(acf)
            }
            
        except Exception as e:
            logger.error(f"Error al analizar patrones de errores: {str(e)}")
            raise
    
    def analyze_feature_importance(self, df: pd.DataFrame) -> Dict:
        """
        Analiza la importancia de las features.
        
        Args:
            df: DataFrame con predicciones
            
        Returns:
            Dict: Resultados del análisis de features
        """
        try:
            # Correlación entre features y error
            df['error'] = df['Importe FX'] - df['predicted_importe_fx']
            df['abs_error'] = abs(df['error'])
            
            # Seleccionar solo columnas numéricas para el análisis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['error', 'abs_error']]
            
            correlations = {}
            for feature in numeric_cols:
                correlations[feature] = float(df[feature].corr(df['abs_error']))
            
            # Gráfico de correlaciones
            plt.figure(figsize=(12, 6))
            correlation_series = pd.Series(correlations)
            correlation_series.sort_values(ascending=True).plot(kind='barh')
            plt.title('Correlación entre Features y Error Absoluto')
            plt.xlabel('Correlación')
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'feature_error_correlation.png')
            plt.close()
            
            return {
                'error_correlations': correlations
            }
            
        except Exception as e:
            logger.error(f"Error al analizar importancia de features: {str(e)}")
            return {}
    
    def generate_performance_report(self) -> None:
        """
        Genera un reporte completo del rendimiento del modelo.
        """
        try:
            logger.info("Iniciando generación de reporte de rendimiento")
            
            # Cargar predicciones
            df = self.load_predictions()
            
            # Realizar análisis
            error_stats = self.analyze_error_distribution(df)
            pattern_analysis = self.analyze_error_patterns(df)
            feature_analysis = self.analyze_feature_importance(df)
            
            # Crear reporte
            report = {
                'timestamp': datetime.now().isoformat(),
                'data_summary': {
                    'n_predictions': len(df),
                    'date_range': {
                        'start': df['prediction_timestamp'].min().isoformat(),
                        'end': df['prediction_timestamp'].max().isoformat()
                    }
                },
                'error_distribution': error_stats,
                'error_patterns': pattern_analysis
            }
            
            # Agregar análisis de features si está disponible
            if feature_analysis:
                report['feature_analysis'] = feature_analysis
            
            # Guardar reporte
            report_file = self.analysis_dir / f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=4)
            
            logger.info(f"Reporte de rendimiento guardado en {report_file}")
            
        except Exception as e:
            logger.error(f"Error al generar reporte de rendimiento: {str(e)}")
            raise

def main():
    """
    Función principal para ejecutar el análisis de rendimiento.
    """
    try:
        analyzer = ModelPerformanceAnalyzer()
        analyzer.generate_performance_report()
        
    except Exception as e:
        logger.error(f"Error en ejecución principal: {str(e)}")
        raise

if __name__ == "__main__":
    main() 