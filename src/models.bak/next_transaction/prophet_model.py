import pandas as pd
import numpy as np
import os
from datetime import datetime
from prophet import Prophet
from ..utils.evaluation import evaluate_next_transaction_model, save_predictions

def load_data(base_path):
    """Cargar datos de entrenamiento y prueba"""
    try:
        # Cargar datos de entrenamiento
        train_path = os.path.join(base_path, 'data', 'processed', 'split_data', 'next_transaction', 'train_data.csv')
        train_data = pd.read_csv(train_path)
        
        # Cargar datos de validación
        val_path = os.path.join(base_path, 'data', 'processed', 'split_data', 'next_transaction', 'val_data.csv')
        val_data = pd.read_csv(val_path)
        
        # Cargar datos de prueba
        test_path = os.path.join(base_path, 'data', 'processed', 'split_data', 'next_transaction', 'test_data.csv')
        test_data = pd.read_csv(test_path)
        
        return train_data, val_data, test_data
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        raise

def prepare_prophet_data(df):
    """Preparar datos para el modelo Prophet"""
    # Crear DataFrame con formato requerido por Prophet
    prophet_data = pd.DataFrame({
        'ds': pd.to_datetime(df['Fecha']),
        'y': df['Días_hasta_siguiente']
    })
    
    # Agregar regresores
    regressors = [
        'Importe FX', 'T/C Cerrado', 'PX_MID', 'PX_HIGH', 'PX_LAST',
        'Importe FX_mean', 'Importe FX_std', 'Importe FX_count',
        'T/C Cerrado_mean', 'T/C Cerrado_std'
    ]
    
    for col in regressors:
        prophet_data[col] = df[col]
    
    return prophet_data

def train_prophet_model(train_data, val_data):
    """Entrenar modelo Prophet"""
    # Configurar modelo
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        seasonality_mode='multiplicative',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    
    # Agregar regresores
    for col in train_data.columns:
        if col not in ['ds', 'y']:
            model.add_regressor(col)
    
    # Entrenar modelo
    model.fit(train_data)
    
    return model

def make_predictions(model, future_data):
    """Realizar predicciones con el modelo Prophet"""
    forecast = model.predict(future_data)
    return forecast['yhat'].values

def save_model(model, base_path):
    """Guardar modelo entrenado"""
    output_path = os.path.join(base_path, 'data', 'processed', 'models', 'prophet', 'prophet_next_transaction_model.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prophet no tiene un método directo de guardado, guardamos los parámetros
    model_params = {
        'changepoint_prior_scale': model.changepoint_prior_scale,
        'seasonality_prior_scale': model.seasonality_prior_scale,
        'seasonality_mode': model.seasonality_mode,
        'daily_seasonality': model.daily_seasonality,
        'weekly_seasonality': model.weekly_seasonality,
        'yearly_seasonality': model.yearly_seasonality
    }
    
    pd.DataFrame([model_params]).to_json(output_path)
    print(f"Parámetros del modelo guardados en: {output_path}")

def main():
    try:
        # Obtener ruta base del proyecto
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Cargar datos
        print("Cargando datos...")
        train_data, val_data, test_data = load_data(base_path)
        
        # Preparar datos para Prophet
        print("Preparando datos para Prophet...")
        train_prophet = prepare_prophet_data(train_data)
        val_prophet = prepare_prophet_data(val_data)
        test_prophet = prepare_prophet_data(test_data)
        
        # Entrenar modelo
        print("Entrenando modelo Prophet...")
        model = train_prophet_model(train_prophet, val_prophet)
        
        # Guardar modelo
        save_model(model, base_path)
        
        # Realizar predicciones
        print("Realizando predicciones...")
        y_pred = make_predictions(model, test_prophet)
        y_test = test_prophet['y'].values
        
        # Evaluar modelo
        print("Evaluando modelo...")
        metrics = evaluate_next_transaction_model(
            model,
            test_prophet.drop(['ds', 'y'], axis=1),
            y_pred,
            base_path,
            'prophet'
        )
        
        # Guardar predicciones
        save_predictions(
            y_test,
            y_pred,
            test_data['Fecha'],
            base_path,
            'prophet',
            'next_transaction'
        )
        
        print("Proceso completado exitosamente")
        print("\nMétricas de evaluación:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
    except Exception as e:
        print(f"Error en la ejecución: {str(e)}")
        raise

if __name__ == "__main__":
    main() 