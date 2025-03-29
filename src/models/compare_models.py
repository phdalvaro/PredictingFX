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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

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

def calculate_mad(series):
    """
    Calcula la Desviación Absoluta Mediana (MAD) de una serie.
    
    Args:
        series: pandas Series
        
    Returns:
        float: MAD value
    """
    median = series.median()
    mad = (series - median).abs().median()
    return mad

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
    
    # Lista de regresores más importantes
    regressors = [
        'Importe FX',
        'Volumen_promedio_diario',
        'Volumen_Ponderado_5',
        'Spread_TC'
    ]
    
    # Normalizar regresores
    for regressor in regressors:
        if regressor in prophet_df.columns:
            # Calcular estadísticas robustas
            q1 = prophet_df[regressor].quantile(0.25)
            q3 = prophet_df[regressor].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Manejar outliers
            prophet_df[regressor] = prophet_df[regressor].clip(lower_bound, upper_bound)
            
            # Normalizar usando robust scaler
            median = prophet_df[regressor].median()
            mad = calculate_mad(prophet_df[regressor])
            prophet_df[regressor] = (prophet_df[regressor] - median) / mad
            
            # Manejar valores nulos
            prophet_df[regressor] = prophet_df[regressor].fillna(0)
    
    # Eliminar columnas que no son regresores ni ds/y
    columns_to_keep = ['ds', 'y'] + regressors
    prophet_df = prophet_df[columns_to_keep]
    
    return prophet_df, regressors

def create_interaction_features(df):
    """
    Crea features de interacción basadas en las features más importantes.
    
    Args:
        df: DataFrame con los datos
        
    Returns:
        DataFrame con nuevas features
    """
    df = df.copy()
    
    # Crear interacciones con Importe FX
    df['Importe_FX_Volumen_Promedio'] = df['Importe FX'] * df['Volumen_promedio_diario']
    df['Importe_FX_Spread'] = df['Importe FX'] * df['Spread_TC']
    
    # Crear interacciones con Volumen_Ponderado_5
    df['Volumen_Ponderado_Spread'] = df['Volumen_Ponderado_5'] * df['Spread_TC']
    
    # Crear ratios
    df['Volumen_Importe_Ratio'] = df['Volumen_diario'] / df['Importe FX']
    df['Spread_Volumen_Ratio'] = df['Spread_TC'] / df['Volumen_diario']
    
    return df

def prepare_data_for_xgboost(df):
    """
    Prepara los datos para XGBoost con feature engineering mejorado.
    
    Args:
        df: DataFrame con los datos
        
    Returns:
        DataFrame preparado para XGBoost
    """
    # Crear una copia del DataFrame
    df_xgb = df.copy()
    
    # Crear features de interacción
    df_xgb = create_interaction_features(df_xgb)
    
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
    
    # Normalizar features
    scaler = StandardScaler()
    df_xgb[numeric_columns] = scaler.fit_transform(df_xgb[numeric_columns])
    
    return df_xgb

def optimize_xgboost(X_train, y_train):
    """
    Optimiza los hiperparámetros de XGBoost usando GridSearchCV.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        
    Returns:
        Mejor modelo XGBoost
    """
    # Definir el espacio de búsqueda de hiperparámetros (reducido)
    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [5, 7],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1]
    }
    
    # Crear el modelo base sin early stopping
    base_model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        eval_metric='rmse'
    )
    
    # Crear el GridSearchCV con menos workers
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,  # Reducir el número de folds
        scoring='neg_root_mean_squared_error',
        n_jobs=1,  # Usar un solo worker
        verbose=1
    )
    
    # Realizar la búsqueda
    grid_search.fit(X_train, y_train)
    
    # Obtener el mejor modelo
    best_model = grid_search.best_estimator_
    
    # Logging de los mejores parámetros
    logging.info("Mejores hiperparámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        logging.info(f"{param}: {value}")
    
    return best_model

def create_stacking_model(X_train, y_train):
    """
    Crea un modelo de stacking combinando múltiples modelos XGBoost.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        
    Returns:
        Modelo de stacking
    """
    # Crear modelos base (reducidos)
    base_models = [
        ('xgb1', XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='rmse'
        )),
        ('xgb2', XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=43,
            eval_metric='rmse'
        ))
    ]
    
    # Crear el modelo final
    final_model = RidgeCV()
    
    # Crear el modelo de stacking con menos workers
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=final_model,
        cv=3,  # Reducir el número de folds
        n_jobs=1  # Usar un solo worker
    )
    
    # Entrenar el modelo
    stacking_model.fit(X_train, y_train)
    
    return stacking_model

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
        prophet = Prophet(
            changepoint_prior_scale=0.001,
            seasonality_prior_scale=0.01,
            holidays_prior_scale=0.01,
            seasonality_mode='multiplicative',
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_range=0.7,
            n_changepoints=15,
            interval_width=0.95
        )
        
        for regressor in regressors:
            prophet.add_regressor(regressor)
        
        prophet.add_country_holidays(country_name='US')
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
        
        # Optimizar y entrenar modelo XGBoost
        logging.info("Optimizando hiperparámetros de XGBoost...")
        xgb_model = optimize_xgboost(X_train, y_train)
        
        # Crear y entrenar modelo de stacking
        logging.info("Entrenando modelo de stacking...")
        stacking_model = create_stacking_model(X_train, y_train)
        
        # Generar pronósticos
        xgb_pred = xgb_model.predict(X_test)
        stacking_pred = stacking_model.predict(X_test)
        
        # Calcular métricas
        prophet_rmse = np.sqrt(mean_squared_error(y_test, prophet_pred))
        prophet_mae = mean_absolute_error(y_test, prophet_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        stacking_rmse = np.sqrt(mean_squared_error(y_test, stacking_pred))
        stacking_mae = mean_absolute_error(y_test, stacking_pred)
        
        # Calcular R²
        prophet_r2 = r2_score(y_test, prophet_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)
        stacking_r2 = r2_score(y_test, stacking_pred)
        
        # Crear DataFrame con resultados
        results = pd.DataFrame({
            'Fecha': test_df['Fecha de cierre'],
            'Real': y_test,
            'Prophet': prophet_pred,
            'XGBoost': xgb_pred,
            'Stacking': stacking_pred
        })
        
        # Crear directorio de resultados si no existe
        os.makedirs('results', exist_ok=True)
        
        # Guardar resultados
        results.to_csv('results/model_comparison.csv', index=False)
        logging.info("Resultados guardados en results/model_comparison.csv")
        
        # Imprimir métricas
        logging.info("\nMétricas de evaluación:")
        logging.info("Prophet - RMSE: %.2f, MAE: %.2f, R²: %.4f", prophet_rmse, prophet_mae, prophet_r2)
        logging.info("XGBoost - RMSE: %.2f, MAE: %.2f, R²: %.4f", xgb_rmse, xgb_mae, xgb_r2)
        logging.info("Stacking - RMSE: %.2f, MAE: %.2f, R²: %.4f", stacking_rmse, stacking_mae, stacking_r2)
        
        # Guardar importancia de features
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        feature_importance.to_csv('results/feature_importance.csv', index=False)
        logging.info("Importancia de features guardada en results/feature_importance.csv")
        
    except Exception as e:
        logging.error("Error en el proceso principal: %s", str(e))
        raise

if __name__ == "__main__":
    main() 