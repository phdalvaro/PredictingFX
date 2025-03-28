import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from xgboost import XGBRegressor, XGBClassifier
from prophet import Prophet
import joblib
import xgboost as xgb

# Obtener ruta base del proyecto
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_dir = os.path.join(base_path, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'train_models.log')),
        logging.StreamHandler()
    ]
)

def load_split_data(base_path, prediction_type):
    """Cargar datos divididos"""
    try:
        split_dir = os.path.join(base_path, 'data', 'processed', 'split')
        
        train_data = pd.read_csv(os.path.join(split_dir, f'{prediction_type}_train.csv'))
        val_data = pd.read_csv(os.path.join(split_dir, f'{prediction_type}_val.csv'))
        test_data = pd.read_csv(os.path.join(split_dir, f'{prediction_type}_test.csv'))
        
        logging.info(f"Datos de {prediction_type} cargados exitosamente")
        return train_data, val_data, test_data
    except Exception as e:
        logging.error(f"Error al cargar datos de {prediction_type}: {str(e)}")
        raise

def prepare_features(df):
    """Preparar características para el entrenamiento"""
    try:
        # Seleccionar características numéricas
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        # Características específicas para cada modelo
        volume_features = [
            'Volumen_Ponderado_5', 'Volumen_promedio_diario', 'Volumen_std_diario',
            'volumen_rolling_7d_mean', 'volumen_rolling_7d_std', 'volumen_rolling_7d_sum',
            'volumen_rolling_14d_mean', 'volumen_rolling_14d_std', 'volumen_rolling_14d_sum',
            'volumen_rolling_30d_mean', 'volumen_rolling_30d_std', 'volumen_rolling_30d_sum',
            'Spread_TC', 'Spread_Ponderado_5', 'TC_std_diario', 'TC_range_diario',
            'tc_rolling_7d_mean', 'tc_rolling_7d_std', 'tc_rolling_14d_mean',
            'tc_rolling_14d_std', 'tc_rolling_30d_mean', 'tc_rolling_30d_std',
            'total_operaciones', 'Frecuencia_Cliente',
            'Hora', 'Minuto', 'Es horario comercial',
            'Volatilidad_Horaria', 'Rango_Horario'
        ]
        
        transaction_features = [
            'Volumen_Ponderado_5', 'Spread_Ponderado_5',
            'Año', 'Mes', 'Día', 'Día de la semana', 'Semana del año',
            'Es fin de semana', 'Es fin de mes', 'Trimestre', 'Mes del año',
            'volumen_rolling_7d_mean', 'volumen_rolling_7d_std', 'volumen_rolling_7d_sum',
            'volumen_rolling_14d_mean', 'volumen_rolling_14d_std', 'volumen_rolling_14d_sum',
            'volumen_rolling_30d_mean', 'volumen_rolling_30d_std', 'volumen_rolling_30d_sum',
            'tc_rolling_7d_mean', 'tc_rolling_7d_std', 'tc_rolling_14d_mean', 'tc_rolling_14d_std',
            'tc_rolling_30d_mean', 'tc_rolling_30d_std', 'TC_std_diario', 'TC_range_diario',
            'Frecuencia_Cliente', 'Categoria_Cliente',
            'Hora', 'Minuto', 'Es horario comercial',
            'Volatilidad_Horaria', 'Rango_Horario'
        ]
        
        # Verificar que todas las características existan
        missing_volume = [col for col in volume_features if col not in df.columns]
        missing_transaction = [col for col in transaction_features if col not in df.columns]
        
        if missing_volume:
            logging.warning(f"Características faltantes para modelo de volumen: {missing_volume}")
        if missing_transaction:
            logging.warning(f"Características faltantes para modelo de transacción: {missing_transaction}")
        
        # Eliminar características faltantes
        volume_features = [col for col in volume_features if col in df.columns]
        transaction_features = [col for col in transaction_features if col in df.columns]
        
        # Guardar lista de características
        save_feature_list(volume_features, 'volume')
        save_feature_list(transaction_features, 'transaction')
        
        return volume_features, transaction_features
    except Exception as e:
        logging.error(f"Error al preparar características: {str(e)}")
        raise

def prepare_data_for_training(df, features, target=None):
    """Preparar datos para el entrenamiento"""
    try:
        # Seleccionar características
        X = df[features].copy()
        
        # Preparar variable objetivo si se proporciona
        y = df[target] if target else None
        
        # Escalar características numéricas
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        
        # Codificar variables categóricas
        label_encoders = {}
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            label_encoders[col] = LabelEncoder()
            X[col] = label_encoders[col].fit_transform(X[col])
        
        return X, y, scaler, label_encoders
    except Exception as e:
        logging.error(f"Error al preparar datos para entrenamiento: {str(e)}")
        raise

def train_volume_model(X_train, X_test, y_train, y_test, feature_cols):
    """Entrenar modelo de predicción de volumen"""
    try:
        # Configurar y entrenar modelo XGBoost
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Obtener importancia de características
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logging.info("Modelo de volumen entrenado exitosamente")
        return model, feature_importance
    except Exception as e:
        logging.error(f"Error al entrenar modelo de volumen: {str(e)}")
        raise

def train_next_transaction_model(X_train, X_test, y_train, y_test, feature_cols):
    """Entrenar modelo de predicción de próxima transacción"""
    try:
        models = {}
        feature_importance = {}
        
        # Entrenar modelo para cada horizonte de predicción
        for target_col in y_train.columns:
            model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            # Entrenar modelo
            model.fit(X_train, y_train[target_col])
            
            # Guardar modelo y su importancia de características
            models[target_col] = model
            feature_importance[target_col] = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        logging.info("Modelos de próxima transacción entrenados exitosamente")
        return models, feature_importance
    except Exception as e:
        logging.error(f"Error al entrenar modelos de próxima transacción: {str(e)}")
        raise

def evaluate_volume_model(model, X_test, y_test):
    """Evaluar modelo de predicción de volumen"""
    try:
        # Realizar predicciones
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        logging.info("Métricas de evaluación del modelo de volumen:")
        for metric, value in metrics.items():
            logging.info(f"{metric.upper()}: {value:.4f}")
        
        return metrics, y_pred
    except Exception as e:
        logging.error(f"Error al evaluar modelo de volumen: {str(e)}")
        raise

def evaluate_next_transaction_models(models, X_test, y_test):
    """Evaluar modelos de predicción de próxima transacción"""
    try:
        results = {}
        predictions = {}
        
        # Evaluar cada modelo
        for target_col, model in models.items():
            # Realizar predicciones
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calcular métricas
            metrics = {
                'accuracy': accuracy_score(y_test[target_col], y_pred),
                'f1': f1_score(y_test[target_col], y_pred)
            }
            
            results[target_col] = metrics
            predictions[target_col] = {
                'pred': y_pred,
                'pred_proba': y_pred_proba
            }
            
            logging.info(f"Métricas de evaluación para {target_col}:")
            for metric, value in metrics.items():
                logging.info(f"{metric.upper()}: {value:.4f}")
        
        return results, predictions
    except Exception as e:
        logging.error(f"Error al evaluar modelos de próxima transacción: {str(e)}")
        raise

def save_models(models, feature_importance, base_path, prediction_type):
    """Guardar modelos y sus métricas"""
    try:
        # Crear directorio para modelos si no existe
        models_dir = os.path.join(base_path, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Guardar modelos
        if isinstance(models, dict):
            for name, model in models.items():
                model_path = os.path.join(models_dir, f'{prediction_type}_{name}.joblib')
                joblib.dump(model, model_path)
                logging.info(f"Modelo guardado en: {model_path}")
        else:
            model_path = os.path.join(models_dir, f'{prediction_type}.joblib')
            joblib.dump(models, model_path)
            logging.info(f"Modelo guardado en: {model_path}")
        
        # Guardar importancia de características
        if isinstance(feature_importance, dict):
            for name, importance in feature_importance.items():
                importance_path = os.path.join(models_dir, f'{prediction_type}_{name}_importance.csv')
                importance.to_csv(importance_path, index=False)
                logging.info(f"Importancia de características guardada en: {importance_path}")
        else:
            importance_path = os.path.join(models_dir, f'{prediction_type}_importance.csv')
            feature_importance.to_csv(importance_path, index=False)
            logging.info(f"Importancia de características guardada en: {importance_path}")
        
    except Exception as e:
        logging.error(f"Error al guardar modelos: {str(e)}")
        raise

def save_feature_list(features, model_type):
    """Guardar lista de características"""
    try:
        # Crear directorio para modelos si no existe
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Guardar lista de características
        feature_path = os.path.join(models_dir, f'{model_type}_features.txt')
        with open(feature_path, 'w', encoding='utf-8') as f:
            for feature in features:
                f.write(f"{feature}\n")
        
        logging.info(f"Lista de características para {model_type} guardada en: {feature_path}")
    except Exception as e:
        logging.error(f"Error al guardar lista de características: {str(e)}")
        raise

def main():
    try:
        # Obtener ruta base del proyecto
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Crear directorio de logs si no existe
        os.makedirs('logs', exist_ok=True)
        
        logging.info("Iniciando entrenamiento de modelos...")
        
        # Cargar datos
        features_path = os.path.join(base_path, 'data', 'processed', 'features.csv')
        df = pd.read_csv(features_path)
        
        # Obtener lista de características
        volume_features, transaction_features = prepare_features(df)
        
        # 1. Entrenar modelo de volumen
        logging.info("\nEntrenando modelo de volumen...")
        
        # Preparar datos para el modelo de volumen
        X, y, scaler, label_encoders = prepare_data_for_training(
            df,
            features=volume_features,
            target='Volumen_diario'
        )
        
        # Dividir datos en entrenamiento y prueba
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Entrenar y evaluar modelo de volumen
        volume_model, volume_importance = train_volume_model(
            X_train, X_test, y_train, y_test, volume_features
        )
        volume_metrics, volume_predictions = evaluate_volume_model(volume_model, X_test, y_test)
        save_models(volume_model, volume_importance, base_path, 'volume')
        
        # 2. Entrenar modelos de próxima transacción
        logging.info("\nEntrenando modelos de próxima transacción...")
        
        # Preparar datos para modelos de transacción
        target_cols = ['Opera_Proximos_7d', 'Opera_Proximos_14d', 'Opera_Proximos_30d']
        X, y, scaler, label_encoders = prepare_data_for_training(
            df,
            features=transaction_features,
            target=target_cols
        )
        
        # Dividir datos en entrenamiento y prueba
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Entrenar y evaluar modelos de transacción
        transaction_models, transaction_importance = train_next_transaction_model(
            X_train, X_test, y_train, y_test, transaction_features
        )
        transaction_metrics, transaction_predictions = evaluate_next_transaction_models(
            transaction_models, X_test, y_test
        )
        save_models(transaction_models, transaction_importance, base_path, 'next_transaction')
        
        logging.info("\nProceso de entrenamiento completado exitosamente")
        
    except Exception as e:
        logging.error(f"Error en el proceso de entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main() 