import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_model(X_train, y_train):
    """
    Entrena un modelo Random Forest.
    
    Args:
        X_train (pd.DataFrame): Características de entrenamiento
        y_train (pd.Series): Valores objetivo de entrenamiento
        
    Returns:
        RandomForestRegressor: Modelo entrenado
    """
    # Crear y entrenar el modelo
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo usando métricas de error.
    
    Args:
        model: Modelo entrenado
        X_test (pd.DataFrame): Características de prueba
        y_test (pd.Series): Valores objetivo de prueba
    """
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nMétricas de evaluación:")
    print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return y_pred

if __name__ == "__main__":
    try:
        # Obtener la ruta base del proyecto
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        processed_dir = os.path.join(base_path, 'data', 'processed')
        
        # Cargar los datos procesados
        print("Cargando datos procesados...")
        X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
        y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).iloc[:, 0]
        y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).iloc[:, 0]
        
        print("\nEntrenando modelo Random Forest...")
        model = train_model(X_train, y_train)
        
        print("\nEvaluando modelo...")
        y_pred = evaluate_model(model, X_test, y_test)
        
        # Guardar el modelo
        model_dir = os.path.join(base_path, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'random_forest_model.joblib')
        joblib.dump(model, model_path)
        
        print(f"\n✅ Modelo guardado en: {model_path}")
        
        # Guardar predicciones
        predictions = pd.DataFrame({
            'Valor Real': y_test,
            'Predicción': y_pred,
            'Error': y_test - y_pred
        })
        predictions.to_csv(os.path.join(processed_dir, 'predictions.csv'), index=False)
        print(f"✅ Predicciones guardadas en: {os.path.join(processed_dir, 'predictions.csv')}")
        
    except Exception as e:
        print(f"\n❌ Error en la ejecución: {str(e)}")
