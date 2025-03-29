import pytest
import pandas as pd
import numpy as np
from src.models.production_model import ProductionModel
from pathlib import Path

@pytest.fixture
def sample_data():
    """Crear datos de prueba."""
    return pd.DataFrame({
        'Importe FX': np.random.normal(1000, 100, 100),
        'Volumen_promedio_diario': np.random.normal(500, 50, 100),
        'Spread_TC': np.random.normal(0.1, 0.01, 100),
        'Volumen_Ponderado_5': np.random.normal(450, 45, 100),
        'Volumen_diario': np.random.normal(480, 48, 100),
        'Fecha de cierre': pd.date_range(start='2024-01-01', periods=100)
    })

@pytest.fixture
def model():
    """Crear instancia del modelo."""
    return ProductionModel(
        model_path='models/xgboost_model.joblib',
        scaler_path='models/scaler.joblib'
    )

def test_validate_data(model, sample_data):
    """Probar validación de datos."""
    is_valid, errors = model.validate_data(sample_data)
    assert is_valid
    assert len(errors) == 0

def test_validate_data_with_missing_columns(model):
    """Probar validación con columnas faltantes."""
    invalid_data = pd.DataFrame({'Importe FX': [1, 2, 3]})
    is_valid, errors = model.validate_data(invalid_data)
    assert not is_valid
    assert len(errors) > 0
    assert 'Columnas faltantes' in errors[0]

def test_validate_data_with_nulls(model):
    """Probar validación con valores nulos."""
    data = pd.DataFrame({
        'Importe FX': [1, 2, None],
        'Volumen_promedio_diario': [1, 2, 3],
        'Spread_TC': [1, 2, 3],
        'Volumen_Ponderado_5': [1, 2, 3],
        'Volumen_diario': [1, 2, 3]
    })
    is_valid, errors = model.validate_data(data)
    assert not is_valid
    assert len(errors) > 0
    assert 'Valores nulos' in errors[0]

def test_validate_data_with_negatives(model):
    """Probar validación con valores negativos."""
    data = pd.DataFrame({
        'Importe FX': [-1, 2, 3],
        'Volumen_promedio_diario': [1, 2, 3],
        'Spread_TC': [1, 2, 3],
        'Volumen_Ponderado_5': [1, 2, 3],
        'Volumen_diario': [1, 2, 3]
    })
    is_valid, errors = model.validate_data(data)
    assert not is_valid
    assert len(errors) > 0
    assert 'Valores negativos' in errors[0]

def test_prepare_features(model, sample_data):
    """Probar preparación de features."""
    features = model.prepare_features(sample_data)
    assert isinstance(features, pd.DataFrame)
    assert 'Importe_FX_Volumen_Promedio' in features.columns
    assert 'Importe_FX_Spread' in features.columns
    assert 'Volumen_Ponderado_Spread' in features.columns

def test_generate_predictions(model, sample_data):
    """Probar generación de predicciones."""
    predictions = model.generate_predictions(sample_data)
    assert isinstance(predictions, pd.DataFrame)
    assert 'Predicción' in predictions.columns
    assert 'Intervalo_Inferior' in predictions.columns
    assert 'Intervalo_Superior' in predictions.columns
    assert 'Confianza' in predictions.columns

def test_get_feature_importance(model):
    """Probar obtención de importancia de features."""
    importance = model.get_feature_importance()
    assert isinstance(importance, pd.DataFrame)
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns
    assert len(importance) > 0

def test_save_predictions(model, sample_data):
    """Probar guardado de predicciones."""
    predictions = model.generate_predictions(sample_data)
    model.save_predictions(predictions)
    assert Path('results/predictions_history.json').exists()

def test_save_metrics(model):
    """Probar guardado de métricas."""
    metrics = {
        'rmse': 100.0,
        'mae': 80.0,
        'r2': 0.95,
        'timestamp': '2024-03-20T10:00:00'
    }
    model.save_metrics(metrics)
    assert Path('results/metrics_history.json').exists()

def test_detect_data_drift(model, sample_data):
    """Probar detección de drift."""
    drift_metrics = model.detect_data_drift(sample_data)
    assert isinstance(drift_metrics, dict)
    for column in sample_data.columns:
        if column in drift_metrics:
            assert 'mean_drift' in drift_metrics[column]
            assert 'std_drift' in drift_metrics[column]
            assert 'drift_detected' in drift_metrics[column]

def test_generate_dashboard(model, sample_data):
    """Probar generación de dashboard."""
    model.generate_predictions(sample_data)
    model.generate_dashboard()
    assert Path('results/dashboard.html').exists() 