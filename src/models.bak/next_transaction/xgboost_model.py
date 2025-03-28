import pandas as pd
import numpy as np
import xgboost as xgb
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from ..utils.evaluation import evaluate_next_transaction_model, save_predictions, plot_feature_importance

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

def prepare_features(df):
    """Preparar características para el modelo"""
    # Seleccionar características
    features = [
        'Importe FX', 'T/C Cerrado', 'PX_MID', 'PX_HIGH', 'PX_LAST',
        'Año', 'Mes', 'Día', 'Día de la semana', 'Trimestre',
        'Importe FX_mean', 'Importe FX_std', 'Importe FX_count',
        'T/C Cerrado_mean', 'T/C Cerrado_std'
    ]
    
    # Agregar características de medias móviles
    for col in ['Importe FX', 'T/C Cerrado', 'PX_MID', 'PX_HIGH', 'PX_LAST']:
        for window in [7, 14, 30]:
            features.extend([
                f'{col}_rolling_mean_{window}',
                f'{col}_rolling_std_{window}',
                f'{col}_diff_{window}',
                f'{col}_pct_change_{window}'
            ])
    
    return df[features], df['Días_hasta_siguiente']

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Entrenar modelo XGBoost"""
    # Configurar parámetros
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # Crear y entrenar modelo
    model = xgb.XGBRegressor(**params)
    
    # Entrenar con early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    return model

def save_model(model, base_path):
    """Guardar modelo entrenado"""
    output_path = os.path.join(base_path, 'data', 'processed', 'models', 'xgboost', 'xgboost_next_transaction_model.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_model(output_path)
    print(f"Modelo guardado en: {output_path}")

def main():
    try:
        # Obtener ruta base del proyecto
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Cargar datos
        print("Cargando datos...")
        train_data, val_data, test_data = load_data(base_path)
        
        # Preparar características
        print("Preparando características...")
        X_train, y_train = prepare_features(train_data)
        X_val, y_val = prepare_features(val_data)
        X_test, y_test = prepare_features(test_data)
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar modelo
        print("Entrenando modelo XGBoost...")
        model = train_xgboost_model(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Guardar modelo
        save_model(model, base_path)
        
        # Realizar predicciones
        print("Realizando predicciones...")
        y_pred = model.predict(X_test_scaled)
        
        # Evaluar modelo
        print("Evaluando modelo...")
        metrics = evaluate_next_transaction_model(
            model,
            X_test_scaled,
            y_pred,
            base_path,
            'xgboost'
        )
        
        # Guardar predicciones
        save_predictions(
            y_test,
            y_pred,
            test_data['Fecha'],
            base_path,
            'xgboost',
            'next_transaction'
        )
        
        # Guardar importancia de características
        plot_feature_importance(
            model,
            X_train.columns,
            base_path,
            'xgboost'
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