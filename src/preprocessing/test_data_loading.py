import pandas as pd
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_loading_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_excel_loading():
    """Prueba la carga del archivo Excel."""
    try:
        # Ruta del archivo
        file_path = Path('data/raw/Fluctua.xlsx')
        
        # Verificar que el archivo existe
        if not file_path.exists():
            logger.error(f"No se encontró el archivo en {file_path}")
            return False
        
        # Intentar leer el archivo
        logger.info(f"Intentando leer archivo: {file_path}")
        df = pd.read_excel(
            file_path,
            sheet_name='OPES',
            skiprows=1,
            engine='openpyxl'
        )
        
        # Mostrar información básica
        logger.info("\nInformación del DataFrame:")
        logger.info(f"Shape: {df.shape}")
        logger.info("\nColumnas:")
        for col in df.columns:
            logger.info(f"- {col}")
        
        logger.info("\nPrimeras 5 filas:")
        logger.info(df.head())
        
        logger.info("\nTipos de datos:")
        logger.info(df.dtypes)
        
        logger.info("\nValores nulos por columna:")
        logger.info(df.isnull().sum())
        
        return True
        
    except Exception as e:
        logger.error(f"Error al cargar datos: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Iniciando prueba de carga de datos")
    success = test_excel_loading()
    
    if success:
        logger.info("Prueba de carga de datos completada exitosamente")
    else:
        logger.error("Prueba de carga de datos falló") 