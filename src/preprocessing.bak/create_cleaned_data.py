import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_sample_data():
    """Crear datos de ejemplo limpios"""
    # Crear fechas
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Crear IDs de clientes
    client_ids = [f'CLI{i:03d}' for i in range(1, 101)]
    
    # Crear datos
    data = []
    for date in dates:
        # Número aleatorio de transacciones por día
        n_transactions = np.random.randint(50, 200)
        
        for _ in range(n_transactions):
            # Seleccionar cliente aleatorio
            client_id = np.random.choice(client_ids)
            
            # Generar hora aleatoria
            hour = np.random.randint(0, 24)
            minute = np.random.randint(0, 60)
            timestamp = date.replace(hour=hour, minute=minute)
            
            # Generar importe FX (entre 1000 y 100000)
            importe = np.random.uniform(1000, 100000)
            
            # Generar tipos de cambio
            tc_cerrado = np.random.uniform(1.05, 1.15)
            px_mid = tc_cerrado - np.random.uniform(0.0001, 0.001)
            
            data.append({
                'Fecha de cierre': timestamp,
                'Codigo del Cliente (IBS)': client_id,
                'Importe FX': importe,
                'T/C Cerrado': tc_cerrado,
                'PX_MID': px_mid
            })
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Ordenar por fecha y cliente
    df = df.sort_values(['Fecha de cierre', 'Codigo del Cliente (IBS)'])
    
    return df

def main():
    """Función principal"""
    try:
        # Obtener ruta base del proyecto
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Crear directorio si no existe
        output_dir = os.path.join(base_path, 'data', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear datos limpios
        df = create_sample_data()
        
        # Guardar archivo
        output_path = os.path.join(output_dir, 'cleaned_data.csv')
        df.to_csv(output_path, index=False)
        
        print(f"Datos limpios generados y guardados en: {output_path}")
        print(f"Total de registros: {len(df)}")
        print(f"Rango de fechas: {df['Fecha de cierre'].min()} a {df['Fecha de cierre'].max()}")
        print(f"Número de clientes únicos: {df['Codigo del Cliente (IBS)'].nunique()}")
        
    except Exception as e:
        print(f"Error al generar datos limpios: {str(e)}")
        raise

if __name__ == "__main__":
    main() 