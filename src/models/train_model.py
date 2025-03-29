import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
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
        logging.FileHandler('logs/train_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """
    Carga y prepara los datos para el entrenamiento del modelo.
    """
    try:
        logger.info("Cargando datos para entrenamiento")
        
        # Cargar datos con características
        df = pd.read_csv('data/processed/features_data.csv', low_memory=False)
        
        # Cargar importancia de características
        feature_importance = pd.read_csv('results/feature_importance.csv')
        top_features = feature_importance.head(15)['Feature'].tolist()
        
        # Preparar características y variable objetivo
        X = df[top_features]
        y = df['Importe FX']
        
        # Escalar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Guardar el scaler
        scaler_path = Path('models/scaler.pkl')
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        
        return X_scaled, y, top_features
        
    except Exception as e:
        logger.error(f"Error al cargar y preparar datos: {str(e)}")
        raise

def optimize_hyperparameters(X: np.ndarray, y: np.ndarray):
    """
    Optimiza los hiperparámetros del modelo usando GridSearchCV.
    """
    try:
        logger.info("Iniciando optimización de hiperparámetros")
        
        # Definir el espacio de búsqueda de hiperparámetros
        param_grid = {
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Crear el modelo base
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Configurar GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Realizar la búsqueda
        grid_search.fit(X, y)
        
        # Guardar resultados
        results_path = Path('results/hyperparameter_optimization.csv')
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = pd.DataFrame(grid_search.cv_results_)
        results.to_csv(results_path, index=False)
        
        logger.info(f"Mejores hiperparámetros: {grid_search.best_params_}")
        logger.info(f"Mejor score: {np.sqrt(-grid_search.best_score_):.2f}")
        
        return grid_search.best_estimator_
        
    except Exception as e:
        logger.error(f"Error en optimización de hiperparámetros: {str(e)}")
        raise

def train_model(X: np.ndarray, y: np.ndarray, best_model: xgb.XGBRegressor):
    """
    Entrena el modelo final y realiza validación cruzada.
    """
    try:
        logger.info("Iniciando entrenamiento del modelo")
        
        # Dividir datos en train y test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entrenar modelo
        best_model.fit(X_train, y_train)
        
        # Realizar predicciones
        y_pred = best_model.predict(X_test)
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info("\nMétricas de rendimiento:")
        logger.info(f"RMSE: {rmse:.2f}")
        logger.info(f"MAE: {mae:.2f}")
        logger.info(f"R2: {r2:.4f}")
        
        # Validación cruzada
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        cv_std = np.sqrt(-cv_scores.std())
        
        logger.info("\nResultados de validación cruzada:")
        logger.info(f"RMSE promedio: {cv_rmse:.2f} (+/- {cv_std:.2f})")
        
        # Guardar modelo
        model_path = Path('models/xgboost_model.pkl')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_path)
        
        return best_model, X_test, y_test, y_pred
        
    except Exception as e:
        logger.error(f"Error en entrenamiento del modelo: {str(e)}")
        raise

def plot_model_results(model: xgb.XGBRegressor, X_test: np.ndarray, y_test: np.ndarray, 
                      y_pred: np.ndarray, feature_names: list):
    """
    Genera gráficos de resultados del modelo.
    """
    try:
        logger.info("Generando gráficos de resultados")
        
        # Crear directorio para gráficos
        plots_dir = Path('results/model_plots')
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Gráfico de valores reales vs predichos
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title('Valores Reales vs Predicciones')
        plt.tight_layout()
        plt.savefig(plots_dir / 'real_vs_predicted.png')
        plt.close()
        
        # Gráfico de residuos
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicciones')
        plt.ylabel('Residuos')
        plt.title('Gráfico de Residuos')
        plt.tight_layout()
        plt.savefig(plots_dir / 'residuals.png')
        plt.close()
        
        # Importancia de características
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance, x='Importance', y='Feature')
        plt.title('Importancia de Características')
        plt.tight_layout()
        plt.savefig(plots_dir / 'feature_importance.png')
        plt.close()
        
        # Guardar importancia de características
        importance.to_csv(plots_dir / 'feature_importance.csv', index=False)
        
    except Exception as e:
        logger.error(f"Error al generar gráficos: {str(e)}")
        raise

def main():
    """
    Función principal para ejecutar el entrenamiento del modelo.
    """
    try:
        # Cargar y preparar datos
        X, y, feature_names = load_and_prepare_data()
        
        # Optimizar hiperparámetros
        best_model = optimize_hyperparameters(X, y)
        
        # Entrenar modelo final
        model, X_test, y_test, y_pred = train_model(X, y, best_model)
        
        # Generar gráficos
        plot_model_results(model, X_test, y_test, y_pred, feature_names)
        
        logger.info("Entrenamiento del modelo completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en ejecución principal: {str(e)}")
        raise

if __name__ == "__main__":
    main() 