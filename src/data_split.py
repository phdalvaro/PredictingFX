import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_data(data, target_column='Unnamed: 24', test_size=0.2):
    """
    Divide el dataset en conjuntos de entrenamiento y prueba.
    Utiliza una estrategia basada en series temporales para mantener el orden cronológico.
    """
    
    # Ordenar por fecha si existe una columna de fecha
    if 'Fecha' in data.columns:
        data['Fecha'] = pd.to_datetime(data['Fecha'], errors='coerce')
        data = data.sort_values('Fecha')

    # Verificar que la columna objetivo existe
    if target_column not in data.columns:
        print(f"\n❌ Error: La columna '{target_column}' no existe en el dataset.")
        print("Columnas disponibles:")
        for col in data.columns:
            print(f"- {col}")
        raise ValueError(f"Columna objetivo '{target_column}' no encontrada")

    # Separar features (X) y target (y)
    X = data.drop([target_column, 'Fecha'], axis=1, errors='ignore')
    y = data[target_column]
    
    # Definir el tamaño del conjunto de prueba (20% de los datos)
    split_index = int(len(data) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"\n✅ Datos divididos correctamente: {len(X_train)} registros para entrenamiento y {len(X_test)} para prueba.")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    try:
        # Obtener la ruta base del proyecto
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Cargar el dataset limpio
        input_file = os.path.join(base_path, 'data', 'cleaned_data.csv')
        print(f"Cargando datos desde: {input_file}")
        
        data = pd.read_csv(input_file)
        print("\nColumnas del dataset limpio:")
        print(data.columns)
        
        # Definir la columna objetivo (usando una columna que sabemos que existe)
        target_column = 'Unnamed: 24'
        
        # Dividir los datos
        X_train, X_test, y_train, y_test = split_data(data, target_column)
        
        # Guardar los conjuntos de datos
        output_dir = os.path.join(base_path, 'data', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
        
        print("\n✅ División completada. Los conjuntos se han guardado en la carpeta '/data/processed'.")
        
    except Exception as e:
        print(f"\n❌ Error en la ejecución: {str(e)}")
