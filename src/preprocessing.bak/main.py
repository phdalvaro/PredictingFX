import os
import sys
from datetime import datetime
import pandas as pd
from preprocessing.feature_engineering import main as feature_engineering
from utils.data_split import main as data_split
from models.volume.xgboost_model import main as train_xgboost_volume
from models.volume.prophet_model import main as train_prophet_volume
from models.next_transaction.xgboost_model import main as train_xgboost_next
from models.next_transaction.prophet_model import main as train_prophet_next
from generate_forecast import main as generate_forecast
from visualization.plot_forecast import main as plot_forecast

def create_log_file(base_path):
    """Crear archivo de registro"""
    log_dir = os.path.join(base_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'execution_{timestamp}.log')
    
    return log_file

def log_step(log_file, step_name, start_time):
    """Registrar paso completado"""
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{step_name} completado en {duration:.2f} segundos")
        f.write(f"\nHora de finalización: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

def main():
    try:
        # Obtener ruta base del proyecto
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Crear archivo de registro
        log_file = create_log_file(base_path)
        start_time = datetime.now()
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Inicio de ejecución: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. Ingeniería de características
        print("\n1. Realizando ingeniería de características...")
        feature_engineering()
        log_step(log_file, "Ingeniería de características", start_time)
        
        # 2. División de datos
        print("\n2. Dividiendo datos en conjuntos de entrenamiento y prueba...")
        data_split()
        log_step(log_file, "División de datos", start_time)
        
        # 3. Entrenamiento de modelos de volumen
        print("\n3. Entrenando modelos de volumen...")
        print("3.1 Entrenando modelo XGBoost de volumen...")
        train_xgboost_volume()
        print("3.2 Entrenando modelo Prophet de volumen...")
        train_prophet_volume()
        log_step(log_file, "Entrenamiento de modelos de volumen", start_time)
        
        # 4. Entrenamiento de modelos de próxima transacción
        print("\n4. Entrenando modelos de próxima transacción...")
        print("4.1 Entrenando modelo XGBoost de próxima transacción...")
        train_xgboost_next()
        print("4.2 Entrenando modelo Prophet de próxima transacción...")
        train_prophet_next()
        log_step(log_file, "Entrenamiento de modelos de próxima transacción", start_time)
        
        # 5. Generación de pronóstico
        print("\n5. Generando pronóstico...")
        generate_forecast()
        log_step(log_file, "Generación de pronóstico", start_time)
        
        # 6. Visualización de resultados
        print("\n6. Generando visualizaciones...")
        plot_forecast()
        log_step(log_file, "Visualización de resultados", start_time)
        
        # Registrar finalización exitosa
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\nProceso completado exitosamente")
            f.write(f"\nDuración total: {total_duration:.2f} segundos")
            f.write(f"\nHora de finalización: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\nProceso completado exitosamente")
        print(f"Registro guardado en: {log_file}")
        
    except Exception as e:
        # Registrar error
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\nError en la ejecución: {str(e)}")
            f.write(f"\nHora del error: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nError en la ejecución: {str(e)}")
        print(f"Registro guardado en: {log_file}")
        sys.exit(1)

if __name__ == "__main__":
    main() 