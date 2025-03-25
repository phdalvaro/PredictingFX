import pandas as pd
import numpy as np
import os

# Función para cargar el dataset
def load_data(file_path, sheet_name='OPES'):
    """Carga el dataset desde un archivo Excel."""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

# Función para limpiar valores faltantes
def clean_data(data):
    """Limpia el dataset tratando los valores faltantes según el contexto."""

    # 1. Identificar valores faltantes
    print("\nValores nulos por columna antes de la limpieza:")
    print(data.isnull().sum())

    # 2. Tratamiento de valores faltantes
    # Ejemplo de decisiones basadas en el análisis previo
    # - Rellenar columnas numéricas con la mediana
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    # - Rellenar columnas de tipo fecha con la fecha anterior (método forward fill)
    if 'Fecha' in data.columns:
        data['Fecha'] = pd.to_datetime(data['Fecha'], errors='coerce')  # Asegurar formato fecha
        data['Fecha'].fillna(method='ffill', inplace=True)

    # - Rellenar columnas categóricas con la moda
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

    # 3. Eliminar filas duplicadas si existen
    data.drop_duplicates(inplace=True)

    # 4. Verificar que no haya valores faltantes restantes
    print("\nValores nulos por columna después de la limpieza:")
    print(data.isnull().sum())

    return data

# Función principal para cargar y limpiar el dataset
if __name__ == "__main__":
    # Obtener la ruta base del proyecto (un nivel arriba del directorio src)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construir las rutas completas
    input_file = os.path.join(base_path, 'data', 'Fluctua.xlsx')
    output_file = os.path.join(base_path, 'data', 'cleaned_data.csv')
    
    print(f"Intentando leer el archivo: {input_file}")
    
    # Cargar datos
    data = load_data(input_file)
    
    # Limpiar datos
    cleaned_data = clean_data(data)
    
    # Guardar el dataset limpio para futuras etapas del proyecto
    cleaned_data.to_csv(output_file, index=False)   
    print(f"\n✅ Limpieza de datos completada. El archivo limpio se ha guardado como '{output_file}'")
