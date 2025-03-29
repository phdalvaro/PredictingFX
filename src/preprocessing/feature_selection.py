import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_selection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """
    Carga y prepara los datos para el análisis de características.
    """
    try:
        logger.info("Cargando datos con características")
        df = pd.read_csv('data/processed/features_data.csv', low_memory=False)
        
        # Convertir fecha a datetime
        df['Fecha de cierre'] = pd.to_datetime(df['Fecha de cierre'])
        
        # Definir características numéricas para análisis
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Eliminar características no relevantes para el análisis
        exclude_features = [
            'Codigo del Cliente (IBS)',
            'Fecha_ordinal',
            'Año',
            'Mes',
            'Día',
            'Día_semana',
            'Trimestre',
            'Equivalente en USD (T/C Cerrado)',
            'Equivalente en USD (T/C Pool)',
            'Utilidad OM-PEN'
        ]
        
        numeric_features = [f for f in numeric_features if f not in exclude_features]
        
        # Asegurar que todas las columnas son numéricas
        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Total de características numéricas: {len(numeric_features)}")
        
        return df, numeric_features
        
    except Exception as e:
        logger.error(f"Error al cargar y preparar datos: {str(e)}")
        raise

def handle_missing_values(df: pd.DataFrame, numeric_features: list) -> pd.DataFrame:
    """
    Maneja los valores faltantes en el DataFrame.
    """
    try:
        logger.info("Manejando valores faltantes")
        
        # Crear una copia del DataFrame
        df_clean = df.copy()
        
        # Reportar valores faltantes antes
        missing_before = df_clean[numeric_features].isnull().sum()
        logger.info("\nValores faltantes antes de la imputación:")
        for col in numeric_features:
            if missing_before[col] > 0:
                logger.info(f"{col}: {missing_before[col]}")
        
        # Imputar valores faltantes con la mediana para cada columna
        imputer = SimpleImputer(strategy='median')
        df_clean[numeric_features] = imputer.fit_transform(df_clean[numeric_features])
        
        # Verificar valores faltantes después
        missing_after = df_clean[numeric_features].isnull().sum()
        logger.info("\nValores faltantes después de la imputación:")
        for col in numeric_features:
            if missing_after[col] > 0:
                logger.info(f"{col}: {missing_after[col]}")
        
        # Verificar que no haya valores infinitos
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean[numeric_features] = df_clean[numeric_features].fillna(df_clean[numeric_features].mean())
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error al manejar valores faltantes: {str(e)}")
        raise

def analyze_feature_importance(df: pd.DataFrame, numeric_features: list):
    """
    Analiza la importancia de las características usando diferentes métodos.
    """
    try:
        logger.info("Analizando importancia de características")
        
        # Preparar datos
        X = df[numeric_features]
        y = df['Importe FX']  # Variable objetivo
        
        # Verificar que no haya valores NaN
        if X.isnull().any().any() or y.isnull().any():
            raise ValueError("Todavía hay valores NaN en los datos")
        
        # Escalar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Método 1: SelectKBest con f_regression
        selector_f = SelectKBest(score_func=f_regression, k='all')
        selector_f.fit(X_scaled, y)
        scores_f = pd.DataFrame({
            'Feature': numeric_features,
            'Score_F': selector_f.scores_
        })
        
        # Método 2: SelectKBest con mutual_info_regression
        selector_mi = SelectKBest(score_func=mutual_info_regression, k='all')
        selector_mi.fit(X_scaled, y)
        scores_mi = pd.DataFrame({
            'Feature': numeric_features,
            'Score_MI': selector_mi.scores_
        })
        
        # Combinar resultados
        scores = pd.merge(scores_f, scores_mi, on='Feature')
        scores['Score_Promedio'] = (scores['Score_F'] + scores['Score_MI']) / 2
        scores = scores.sort_values('Score_Promedio', ascending=False)
        
        # Guardar resultados
        output_path = Path('results/feature_importance.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(output_path, index=False)
        logger.info(f"Resultados guardados en {output_path}")
        
        return scores
        
    except Exception as e:
        logger.error(f"Error al analizar importancia de características: {str(e)}")
        raise

def plot_feature_importance(df: pd.DataFrame, scores: pd.DataFrame):
    """
    Genera gráficos de importancia de características.
    """
    try:
        logger.info("Generando gráficos de importancia de características")
        
        # Crear directorio para gráficos
        plots_dir = Path('results/plots')
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Gráfico de barras para las top 15 características
        plt.figure(figsize=(12, 6))
        sns.barplot(data=scores.head(15), x='Score_Promedio', y='Feature')
        plt.title('Top 15 Características más Importantes')
        plt.tight_layout()
        plt.savefig(plots_dir / 'top_15_features.png')
        plt.close()
        
        # Gráfico de correlación para las top 10 características
        top_features = scores.head(10)['Feature'].tolist()
        correlation_matrix = df[top_features + ['Importe FX']].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlación - Top 10 Características')
        plt.tight_layout()
        plt.savefig(plots_dir / 'correlation_matrix.png')
        plt.close()
        
        logger.info("\nTop 10 características más importantes:")
        for idx, row in scores.head(10).iterrows():
            logger.info(f"{row['Feature']}: {row['Score_Promedio']:.4f}")
        
    except Exception as e:
        logger.error(f"Error al generar gráficos: {str(e)}")
        raise

def analyze_feature_distributions(df: pd.DataFrame, top_features: list):
    """
    Analiza las distribuciones de las características más importantes.
    """
    try:
        logger.info("Analizando distribuciones de características")
        
        # Crear directorio para análisis
        analysis_dir = Path('results/feature_analysis')
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Análisis de distribuciones
        for feature in top_features:
            # Estadísticas básicas
            stats = df[feature].describe()
            
            # Gráfico de distribución
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=feature, bins=50)
            plt.title(f'Distribución de {feature}')
            plt.tight_layout()
            plt.savefig(analysis_dir / f'distribution_{feature}.png')
            plt.close()
            
            # Guardar estadísticas
            stats.to_csv(analysis_dir / f'stats_{feature}.csv')
            
            logger.info(f"\nAnálisis de {feature}:")
            logger.info(f"Media: {stats['mean']:.2f}")
            logger.info(f"Desviación estándar: {stats['std']:.2f}")
            logger.info(f"Mínimo: {stats['min']:.2f}")
            logger.info(f"Máximo: {stats['max']:.2f}")
            logger.info(f"Valores nulos: {df[feature].isnull().sum()}")
            
    except Exception as e:
        logger.error(f"Error al analizar distribuciones: {str(e)}")
        raise

def main():
    """
    Función principal para ejecutar el análisis de características.
    """
    try:
        # Cargar y preparar datos
        df, numeric_features = load_and_prepare_data()
        
        # Manejar valores faltantes
        df_clean = handle_missing_values(df, numeric_features)
        
        # Analizar importancia de características
        scores = analyze_feature_importance(df_clean, numeric_features)
        
        # Generar gráficos
        plot_feature_importance(df_clean, scores)
        
        # Analizar distribuciones de las top 10 características
        top_features = scores.head(10)['Feature'].tolist()
        analyze_feature_distributions(df_clean, top_features)
        
        logger.info("Análisis de características completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en ejecución principal: {str(e)}")
        raise

if __name__ == "__main__":
    main() 