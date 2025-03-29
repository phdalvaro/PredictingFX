import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
from src.models.production_model import ProductionModel
from src.data.data_processor import DataProcessor
from src.data.feature_engineering import FeatureEngineer

@pytest.fixture
def sample_raw_data():
    """Crea datos de muestra para pruebas."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'Fecha de cierre': dates,
        'Importe FX': np.random.uniform(1000, 10000, 100),
        'Volumen_promedio_diario': np.random.uniform(100, 1000, 100),
        'Spread_TC': np.random.uniform(0.01, 0.1, 100),
        'Volumen_Ponderado_5': np.random.uniform(100, 1000, 100),
        'Volumen_diario': np.random.uniform(100, 1000, 100),
        'Codigo del Cliente (IBS)': np.random.randint(1000, 9999, 100),
        'Mes del año': dates.month,
        'Día de la semana nombre': dates.day_name(),
        'Categoria_Cliente': np.random.choice(['A', 'B', 'C'], 100),
        'Frecuencia_Cliente': np.random.randint(1, 10, 100),
        'Proxima_Fecha': dates + timedelta(days=7)
    })

@pytest.fixture
def model():
    """Crea una instancia del modelo para pruebas."""
    return ProductionModel(
        model_path='models/xgboost_model.joblib',
        scaler_path='models/scaler.joblib'
    )

def test_data_cleaning_pipeline(sample_raw_data):
    """Prueba el pipeline completo de limpieza de datos."""
    processor = DataProcessor()
    cleaned_data = processor.clean_data(sample_raw_data)
    
    assert not cleaned_data.isnull().any().any()
    assert not (cleaned_data < 0).any().any()
    assert len(cleaned_data) == len(sample_raw_data)

def test_feature_creation_pipeline(sample_raw_data):
    """Prueba el pipeline completo de creación de features."""
    processor = DataProcessor()
    engineer = FeatureEngineer()
    
    cleaned_data = processor.clean_data(sample_raw_data)
    features = engineer.create_features(cleaned_data)
    
    expected_features = [
        'Importe_FX_Volumen_Promedio',
        'Importe_FX_Spread',
        'Volumen_Ponderado_Spread',
        'Volumen_Importe_Ratio',
        'Spread_Volumen_Ratio'
    ]
    
    for feature in expected_features:
        assert feature in features.columns

def test_full_model_pipeline(sample_raw_data, model):
    """Prueba el pipeline completo del modelo desde datos crudos hasta predicciones."""
    # Preparar datos
    processor = DataProcessor()
    cleaned_data = processor.clean_data(sample_raw_data)
    
    # Generar predicciones
    predictions = model.generate_predictions(cleaned_data)
    
    assert 'Predicción' in predictions.columns
    assert 'Intervalo_Inferior' in predictions.columns
    assert 'Intervalo_Superior' in predictions.columns
    assert 'Confianza' in predictions.columns
    assert len(predictions) == len(cleaned_data)

def test_model_monitoring_pipeline(sample_raw_data, model):
    """Prueba el pipeline completo de monitoreo del modelo."""
    # Detectar drift
    drift_metrics = model.detect_data_drift(sample_raw_data)
    
    # Generar predicciones
    predictions = model.generate_predictions(sample_raw_data)
    
    # Guardar predicciones
    model.save_predictions(predictions)
    
    # Generar dashboard
    model.generate_dashboard()
    
    # Verificar archivos generados
    assert Path('results/predictions_history.json').exists()
    assert Path('results/dashboard.html').exists()
    assert Path('results/feature_stats_history.json').exists()

def test_model_metrics_pipeline(sample_raw_data, model):
    """Prueba el pipeline completo de métricas del modelo."""
    # Generar predicciones
    predictions = model.generate_predictions(sample_raw_data)
    
    # Evaluar predicciones
    actual_values = sample_raw_data['Volumen_diario']
    metrics = model.evaluate_predictions(predictions, actual_values)
    
    # Verificar métricas
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert 'timestamp' in metrics
    
    # Verificar archivo de métricas
    assert Path('results/metrics_history.json').exists()

def test_model_feature_importance_pipeline(sample_raw_data, model):
    """Prueba el pipeline completo de importancia de features."""
    # Obtener importancia de features
    importance = model.get_feature_importance()
    
    # Verificar estructura
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns
    assert len(importance) > 0
    
    # Verificar archivo de importancia
    assert Path('results/feature_importance_production.csv').exists()

def test_model_security_pipeline(sample_raw_data, model):
    """Prueba el pipeline completo de seguridad del modelo."""
    # Verificar carga segura del modelo
    assert model.model is not None
    assert model.scaler is not None
    
    # Verificar archivos de hash
    assert Path('models/xgboost_model.joblib.hash').exists()
    assert Path('models/scaler.joblib.hash').exists()
    
    # Verificar directorio de seguridad
    assert Path('security/key.key').exists()

def test_model_data_validation_pipeline(sample_raw_data, model):
    """Prueba el pipeline completo de validación de datos."""
    # Validar datos
    is_valid, errors = model.validate_data(sample_raw_data)
    
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)
    
    # Verificar que los datos son válidos
    assert is_valid
    assert len(errors) == 0

def test_model_error_handling_pipeline():
    """Prueba el manejo de errores en el pipeline completo."""
    # Intentar cargar modelo con ruta incorrecta
    with pytest.raises(FileNotFoundError):
        ProductionModel(
            model_path='models/nonexistent_model.joblib',
            scaler_path='models/nonexistent_scaler.joblib'
        )
    
    # Intentar validar datos inválidos
    invalid_data = pd.DataFrame()
    model = ProductionModel(
        model_path='models/xgboost_model.joblib',
        scaler_path='models/scaler.joblib'
    )
    
    is_valid, errors = model.validate_data(invalid_data)
    assert not is_valid
    assert len(errors) > 0

def test_model_performance_pipeline(sample_raw_data, model):
    """Prueba el rendimiento del pipeline completo."""
    import time
    
    # Medir tiempo de procesamiento
    start_time = time.time()
    
    # Ejecutar pipeline completo
    predictions = model.generate_predictions(sample_raw_data)
    drift_metrics = model.detect_data_drift(sample_raw_data)
    model.generate_dashboard()
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Verificar que el tiempo de procesamiento es razonable
    assert processing_time < 10  # menos de 10 segundos 