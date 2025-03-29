import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_path: str = "data/raw/Fluctua.xlsx"):
        """
        Inicializa el procesador de datos.
        
        Args:
            data_path: Ruta al archivo de datos
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None
        self.validation_report = {}
        
    def load_data(self) -> None:
        """
        Carga los datos del archivo Excel.
        """
        try:
            logger.info(f"Cargando datos desde {self.data_path}")
            
            # Primero, leer la primera fila para verificar
            first_row = pd.read_excel(self.data_path, sheet_name="OPES", nrows=1)
            logger.info("\nPrimera fila del archivo:")
            for col in first_row.columns:
                logger.info(f"- {col}")
            
            # Leer la segunda fila como encabezados
            self.raw_data = pd.read_excel(
                self.data_path,
                sheet_name="OPES",
                header=1  # Usar la segunda fila como encabezados
            )
            
            logger.info(f"\nDatos cargados exitosamente. Shape: {self.raw_data.shape}")
            logger.info("\nEncabezados (segunda fila):")
            for col in self.raw_data.columns:
                logger.info(f"- {col}")
            
            # Verificar columnas críticas
            critical_cols = ["Fecha de cierre", "Importe FX"]
            for col in critical_cols:
                if col in self.raw_data.columns:
                    logger.info(f"✓ Columna crítica encontrada: {col}")
                else:
                    logger.error(f"✗ Columna crítica no encontrada: {col}")
            
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            raise
            
    def validate_data(self) -> Dict:
        """
        Realiza validaciones básicas de los datos.
        
        Returns:
            Dict: Reporte de validación
        """
        validation_report = {
            "total_rows": len(self.raw_data),
            "missing_values": {},
            "data_types": {},
            "unique_values": {},
            "issues": []
        }
        
        # Validar valores faltantes
        missing_values = self.raw_data.isnull().sum()
        validation_report["missing_values"] = missing_values[missing_values > 0].to_dict()
        
        # Validar tipos de datos
        validation_report["data_types"] = self.raw_data.dtypes.astype(str).to_dict()
        
        # Validar valores únicos
        for col in self.raw_data.columns:
            unique_count = self.raw_data[col].nunique()
            validation_report["unique_values"][col] = int(unique_count)
            
        # Validaciones específicas
        if "Importe FX" not in self.raw_data.columns:
            validation_report["issues"].append("Columna 'Importe FX' no encontrada")
            
        if "Fecha" not in self.raw_data.columns:
            validation_report["issues"].append("Columna 'Fecha' no encontrada")
            
        self.validation_report = validation_report
        return validation_report
        
    def clean_data(self) -> pd.DataFrame:
        """
        Limpia los datos realizando las siguientes operaciones:
        1. Elimina duplicados
        2. Maneja valores faltantes
        3. Corrige tipos de datos
        4. Elimina outliers
        5. Elimina columnas sin nombre
        """
        try:
            logger.info("Iniciando limpieza de datos")
            
            # Crear copia para no modificar datos originales
            df = self.raw_data.copy()
            
            # Eliminar columnas sin nombre
            unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
            if unnamed_cols:
                logger.info(f"Eliminando columnas sin nombre: {unnamed_cols}")
                df = df.drop(columns=unnamed_cols)
            
            # Eliminar duplicados
            initial_rows = len(df)
            df = df.drop_duplicates()
            logger.info(f"Duplicados eliminados: {initial_rows - len(df)}")
            
            # Identificar y procesar columna de fecha
            date_col = "Fecha de cierre"
            if date_col in df.columns:
                logger.info(f"Procesando columna de fecha: {date_col}")
                df[date_col] = pd.to_datetime(df[date_col])
            else:
                logger.warning("No se encontró columna de fecha esperada")
                
            # Identificar y procesar columna de importe
            fx_col = "Importe FX"
            if fx_col in df.columns:
                logger.info(f"Procesando columna de importe: {fx_col}")
                # Convertir a numérico, reemplazando errores con NaN
                df[fx_col] = pd.to_numeric(df[fx_col], errors='coerce')
                
                # Reportar valores no numéricos encontrados
                non_numeric = df[fx_col].isna().sum()
                if non_numeric > 0:
                    logger.warning(f"Se encontraron {non_numeric} valores no numéricos en {fx_col}")
                
                # Manejar valores faltantes
                if df[fx_col].isnull().any():
                    # Calcular mediana por día de la semana
                    df["Dia_Semana"] = df[date_col].dt.dayofweek
                    median_by_day = df.groupby("Dia_Semana")[fx_col].transform("median")
                    df[fx_col] = df[fx_col].fillna(median_by_day)
                    logger.info(f"Valores faltantes en {fx_col} completados con mediana por día")
                
                # Eliminar outliers
                Q1 = df[fx_col].quantile(0.25)
                Q3 = df[fx_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[
                    (df[fx_col] < lower_bound) | 
                    (df[fx_col] > upper_bound)
                ]
                
                if len(outliers) > 0:
                    logger.info(f"Outliers encontrados en {fx_col}: {len(outliers)} registros")
                    logger.info(f"Rango de valores antes de eliminar outliers: [{df[fx_col].min():.2f}, {df[fx_col].max():.2f}]")
                
                df = df[
                    (df[fx_col] >= lower_bound) & 
                    (df[fx_col] <= upper_bound)
                ]
                
                logger.info(f"Rango de valores después de eliminar outliers: [{df[fx_col].min():.2f}, {df[fx_col].max():.2f}]")
            else:
                logger.error(f"No se encontró la columna {fx_col}")
                raise ValueError(f"Columna requerida no encontrada: {fx_col}")
                
            # Procesar otras columnas numéricas importantes
            numeric_cols = [
                "T/C Cerrado",
                "TC CONTRAPARTE",
                "PX_MID",
                "PX_HIGH",
                "PX_LAST",
                "Monto Cliente",
                "Equivalente en USD (T/C Cerrado)",
                "Equivalente en USD (T/C Pool)",
                "Monto Contraparte",
                "Spread\n"  # Corregido el nombre de la columna
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].median())
                        logger.info(f"Valores faltantes en {col} completados con mediana")
            
            # Renombrar columnas con nombres problemáticos
            df = df.rename(columns={
                "Spread\n": "Spread"
            })
                
            self.processed_data = df
            logger.info(f"Datos limpios. Shape final: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error en la limpieza de datos: {str(e)}")
            raise
            
    def save_processed_data(self, output_path: str = "data/processed/processed_data.csv") -> None:
        """
        Guarda los datos procesados.
        
        Args:
            output_path: Ruta donde guardar los datos procesados
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.processed_data.to_csv(output_path, index=False)
            logger.info(f"Datos procesados guardados en {output_path}")
            
            # Guardar reporte de validación
            validation_report_path = output_path.parent / "validation_report.json"
            with open(validation_report_path, "w") as f:
                json.dump(self.validation_report, f, indent=4)
            logger.info(f"Reporte de validación guardado en {validation_report_path}")
            
        except Exception as e:
            logger.error(f"Error al guardar datos procesados: {str(e)}")
            raise
            
    def process_pipeline(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Ejecuta el pipeline completo de procesamiento de datos.
        
        Returns:
            Tuple[pd.DataFrame, Dict]: Datos procesados y reporte de validación
        """
        self.load_data()
        validation_report = self.validate_data()
        processed_data = self.clean_data()
        self.save_processed_data()
        
        return processed_data, validation_report

if __name__ == "__main__":
    # Ejemplo de uso
    processor = DataProcessor()
    processed_data, validation_report = processor.process_pipeline()
    
    print("\nReporte de Validación:")
    print(json.dumps(validation_report, indent=2))
    
    print("\nPrimeras filas de datos procesados:")
    print(processed_data.head()) 