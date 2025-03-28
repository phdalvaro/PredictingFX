import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os

def calculate_metrics(y_true, y_pred):
    """Calcular métricas de evaluación"""
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics

def evaluate_volume_model(model, X_test, y_test, base_path, model_name):
    """Evaluar modelo de predicción de volumen"""
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    metrics = calculate_metrics(y_test, y_pred)
    
    # Guardar resultados
    results = {
        'Modelo': model_name,
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'R2': metrics['R2']
    }
    
    # Guardar en CSV
    output_path = os.path.join(base_path, 'data', 'processed', 'models', 'evaluation', f'{model_name}_metrics.csv')
    pd.DataFrame([results]).to_csv(output_path, index=False)
    
    return metrics

def evaluate_next_transaction_model(model, X_test, y_test, base_path, model_name):
    """Evaluar modelo de predicción de próxima transacción"""
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    metrics = calculate_metrics(y_test, y_pred)
    
    # Guardar resultados
    results = {
        'Modelo': model_name,
        'RMSE_días': metrics['RMSE'],
        'MAE_días': metrics['MAE'],
        'R2': metrics['R2']
    }
    
    # Guardar en CSV
    output_path = os.path.join(base_path, 'data', 'processed', 'models', 'evaluation', f'{model_name}_metrics.csv')
    pd.DataFrame([results]).to_csv(output_path, index=False)
    
    return metrics

def save_predictions(y_true, y_pred, dates, base_path, model_name, prediction_type):
    """Guardar predicciones y valores reales"""
    # Crear DataFrame con resultados
    results = pd.DataFrame({
        'Fecha': dates,
        'Valor_Real': y_true,
        'Predicción': y_pred,
        'Error': y_true - y_pred,
        'Error_Absoluto': np.abs(y_true - y_pred)
    })
    
    # Guardar en CSV
    output_path = os.path.join(base_path, 'data', 'processed', 'predictions', f'{model_name}_{prediction_type}_predictions.csv')
    results.to_csv(output_path, index=False)
    
    print(f"Predicciones guardadas en: {output_path}")

def plot_feature_importance(model, feature_names, base_path, model_name):
    """Guardar importancia de características"""
    # Obtener importancia de características
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'get_score'):
        importance = [model.get_score().get(f'f{i}', 0) for i in range(len(feature_names))]
    else:
        print("Modelo no tiene atributo de importancia de características")
        return
    
    # Crear DataFrame
    importance_df = pd.DataFrame({
        'Característica': feature_names,
        'Importancia': importance
    }).sort_values('Importancia', ascending=False)
    
    # Guardar en CSV
    output_path = os.path.join(base_path, 'data', 'processed', 'models', 'evaluation', f'{model_name}_feature_importance.csv')
    importance_df.to_csv(output_path, index=False)
    
    print(f"Importancia de características guardada en: {output_path}") 