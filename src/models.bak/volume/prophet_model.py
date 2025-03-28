import pandas as pd
import numpy as np
from prophet import Prophet
import os
from datetime import datetime
from ..utils.evaluation import evaluate_volume_model, save_predictions

def load_data(base_path):
    """Cargar datos de entrenamiento y prueba"""
    try:
        # Cargar datos de entrenamiento
        train_path = os.path.join(base_path, 'data', 'processed', 'split_data', 'volume', 'train_data.csv')
        train_data = pd.read_csv(train_path)
        
        # Cargar datos de validación
        val_path = os.path.join(base_path, 'data', 'processed', 'split_data', 'volume', 'val_data.csv')
        val_data = pd.read_csv(val_path)
        
        # Cargar datos de prueba
        test_path = os.path.join(base_path, 'data', 'processed', 'split_data', 'volume', 'test_data.csv')
        test_data = pd.read_csv(test_path)
        
        return train_data, val_data, test_data
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        raise

def prepare_prophet_data(df):
    """Preparar datos para Prophet"""
    # Prophet requiere columnas 'ds' y 'y'
    prophet_data = df[['Fecha', 'Volumen_diario']].copy()
    prophet_data.columns = ['ds', 'y']
    return prophet_data

def train_prophet_model(train_data, val_data):
    """Entrenar modelo Prophet"""
    # Configurar modelo
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.95
    )
    
    # Agregar regresores
    for col in ['T/C Cerrado', 'PX_MID', 'PX_HIGH', 'PX_LAST']:
        model.add_regressor(col)
    
    # Preparar datos de entrenamiento
    train_prophet = prepare_prophet_data(train_data)
    
    # Agregar regresores a datos de entrenamiento
    for col in ['T/C Cerrado', 'PX_MID', 'PX_HIGH', 'PX_LAST']:
        train_prophet[col] = train_data[col]
    
    # Entrenar modelo
    model.fit(train_prophet)
    
    return model

def make_predictions(model, future_data):
    """Realizar predicciones con Prophet"""
    # Preparar datos futuros
    future = prepare_prophet_data(future_data)
    
    # Agregar regresores
    for col in ['T/C Cerrado', 'PX_MID', 'PX_HIGH', 'PX_LAST']:
        future[col] = future_data[col]
    
    # Realizar predicciones
    forecast = model.predict(future)
    
    return forecast

def save_model(model, base_path):
    """Guardar modelo entrenado"""
    output_path = os.path.join(base_path, 'data', 'processed', 'models', 'prophet', 'prophet_model.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"Modelo guardado en: {output_path}")

def main():
    try:
        # Obtener ruta base del proyecto
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Cargar datos
        print("Cargando datos...")
        train_data, val_data, test_data = load_data(base_path)
        
        # Entrenar modelo
        print("Entrenando modelo Prophet...")
        model = train_prophet_model(train_data, val_data)
        
        # Guardar modelo
        save_model(model, base_path)
        
        # Realizar predicciones en conjunto de prueba
        print("Realizando predicciones...")
        forecast = make_predictions(model, test_data)
        
        # Evaluar modelo
        print("Evaluando modelo...")
        metrics = evaluate_volume_model(
            model,
            test_data,
            forecast['yhat'],
            base_path,
            'prophet'
        )
        
        # Guardar predicciones
        save_predictions(
            test_data['Volumen_diario'],
            forecast['yhat'],
            test_data['Fecha'],
            base_path,
            'prophet',
            'volume'
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