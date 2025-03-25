import pandas as pd
import numpy as np
import os

# Función para cargar el dataset
def load_data(file_path, sheet_name=None):
    """Carga el dataset desde un archivo Excel."""
    try:
        # Leer todas las hojas disponibles
        excel_file = pd.ExcelFile(file_path)
        available_sheets = excel_file.sheet_names
        print(f"\nHojas disponibles en el archivo: {available_sheets}")
        
        # Si no se especifica una hoja, usar la primera
        if sheet_name is None:
            sheet_name = available_sheets[0]
            print(f"Usando la primera hoja: {sheet_name}")
        
        # Intentar cargar la hoja especificada
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"\nDatos cargados exitosamente de la hoja: {sheet_name}")
        print(f"Dimensiones del dataset: {data.shape}")
        return data
    except Exception as e:
        print(f"\n❌ Error al cargar el archivo: {str(e)}")
        raise

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
    if len(categorical_cols) > 0:
        try:
            data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
        except Exception as e:
            print(f"Advertencia al rellenar valores categóricos: {str(e)}")

    # 3. Eliminar filas duplicadas si existen
    data.drop_duplicates(inplace=True)

    # 4. Verificar que no haya valores faltantes restantes
    print("\nValores nulos por columna después de la limpieza:")
    print(data.isnull().sum())

    return data

# Función principal para cargar y limpiar el dataset
if __name__ == "__main__":
    try:
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
        
    except Exception as e:
        print(f"\n❌ Error en la ejecución: {str(e)}")
