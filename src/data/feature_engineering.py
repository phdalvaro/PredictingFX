import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
import json
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, data_path: str = "data/processed/processed_data.csv"):
        """
        Inicializa el ingeniero de features.
        
        Args:
            data_path: Ruta al archivo de datos procesados
        """
        self.data_path = Path(data_path)
        self.data = None
        self.feature_report = {}
        
    def load_data(self) -> None:
        """
        Carga los datos procesados.
        """
        try:
            logger.info(f"Cargando datos procesados desde {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            self.data['Fecha de cierre'] = pd.to_datetime(self.data['Fecha de cierre'])
            logger.info(f"Datos cargados exitosamente. Shape: {self.data.shape}")
            
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            raise
            
    def create_time_features(self) -> None:
        """
        Crea features basadas en el tiempo.
        """
        try:
            logger.info("Creando features temporales")
            
            # Extraer componentes de la fecha
            self.data['Año'] = self.data['Fecha de cierre'].dt.year
            self.data['Mes'] = self.data['Fecha de cierre'].dt.month
            self.data['Dia'] = self.data['Fecha de cierre'].dt.day
            self.data['Dia_Semana'] = self.data['Fecha de cierre'].dt.dayofweek
            self.data['Dia_Nombre'] = self.data['Fecha de cierre'].dt.day_name()
            self.data['Es_Finde'] = self.data['Dia_Semana'].isin([5, 6]).astype(int)
            
            # Crear features de temporada
            self.data['Es_Inicio_Mes'] = (self.data['Dia'] <= 5).astype(int)
            self.data['Es_Fin_Mes'] = (self.data['Dia'] >= 25).astype(int)
            
            # Crear features de tendencia
            self.data['Dias_Desde_Inicio'] = (self.data['Fecha de cierre'] - self.data['Fecha de cierre'].min()).dt.days
            
            logger.info("Features temporales creadas exitosamente")
            
        except Exception as e:
            logger.error(f"Error al crear features temporales: {str(e)}")
            raise
            
    def create_rolling_features(self) -> None:
        """
        Crea features basadas en ventanas móviles.
        """
        try:
            logger.info("Creando features de ventanas móviles")
            
            # Ordenar por fecha
            self.data = self.data.sort_values('Fecha de cierre')
            
            # Ventanas móviles para Importe FX
            for window in [7, 14, 30]:
                # Media móvil
                self.data[f'Importe_FX_Media_{window}d'] = self.data.groupby('Codigo del Cliente (IBS)')['Importe FX'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                # Desviación estándar móvil
                self.data[f'Importe_FX_Std_{window}d'] = self.data.groupby('Codigo del Cliente (IBS)')['Importe FX'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                # Máximo móvil
                self.data[f'Importe_FX_Max_{window}d'] = self.data.groupby('Codigo del Cliente (IBS)')['Importe FX'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
                # Mínimo móvil
                self.data[f'Importe_FX_Min_{window}d'] = self.data.groupby('Codigo del Cliente (IBS)')['Importe FX'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
            
            # Ventanas móviles para T/C
            for window in [7, 14, 30]:
                self.data[f'TC_Media_{window}d'] = self.data['T/C Cerrado'].rolling(window=window, min_periods=1).mean()
                self.data[f'TC_Std_{window}d'] = self.data['T/C Cerrado'].rolling(window=window, min_periods=1).std()
            
            logger.info("Features de ventanas móviles creadas exitosamente")
            
        except Exception as e:
            logger.error(f"Error al crear features de ventanas móviles: {str(e)}")
            raise
            
    def create_categorical_features(self) -> None:
        """
        Crea features basadas en variables categóricas.
        """
        try:
            logger.info("Creando features categóricas")
            
            # Codificar variables categóricas
            categorical_cols = [
                'Rubro de Negocio',
                'PN/PJ',
                'Ambito de Negocio',
                'Producto',
                'Tipo de Operación',
                'Moneda FX',
                'Tipo de Cliente'
            ]
            
            for col in categorical_cols:
                if col in self.data.columns:
                    # Crear dummies
                    dummies = pd.get_dummies(self.data[col], prefix=col)
                    self.data = pd.concat([self.data, dummies], axis=1)
                    logger.info(f"Dummies creadas para {col}")
            
            logger.info("Features categóricas creadas exitosamente")
            
        except Exception as e:
            logger.error(f"Error al crear features categóricas: {str(e)}")
            raise
            
    def create_interaction_features(self) -> None:
        """
        Crea features de interacción entre variables.
        """
        try:
            logger.info("Creando features de interacción")
            
            # Interacción entre Importe FX y T/C
            self.data['Importe_FX_TC'] = self.data['Importe FX'] * self.data['T/C Cerrado']
            
            # Interacción entre Spread y Importe FX
            self.data['Spread_Importe'] = self.data['Spread'] * self.data['Importe FX']
            
            # Interacción entre tipo de operación y monto
            self.data['Operacion_Monto'] = self.data['Tipo de Operación'].map({'Compra': 1, 'Venta': -1}) * self.data['Importe FX']
            
            logger.info("Features de interacción creadas exitosamente")
            
        except Exception as e:
            logger.error(f"Error al crear features de interacción: {str(e)}")
            raise
            
    def generate_feature_report(self) -> Dict:
        """
        Genera un reporte de las features creadas.
        """
        feature_report = {
            "total_features": len(self.data.columns),
            "feature_types": {
                "numeric": len(self.data.select_dtypes(include=[np.number]).columns),
                "categorical": len(self.data.select_dtypes(include=['object']).columns),
                "datetime": len(self.data.select_dtypes(include=['datetime64']).columns)
            },
            "features_by_category": {
                "time_features": [col for col in self.data.columns if any(x in col for x in ['Año', 'Mes', 'Dia', 'Semana'])],
                "rolling_features": [col for col in self.data.columns if any(x in col for x in ['Media', 'Std', 'Max', 'Min'])],
                "categorical_features": [col for col in self.data.columns if any(x in col for x in ['Rubro', 'PN', 'Ambito', 'Producto', 'Tipo'])],
                "interaction_features": [col for col in self.data.columns if any(x in col for x in ['_TC', 'Spread_', 'Operacion_'])]
            }
        }
        
        self.feature_report = feature_report
        return feature_report
        
    def save_features(self, output_path: str = "data/processed/features.csv") -> None:
        """
        Guarda los datos con las nuevas features.
        
        Args:
            output_path: Ruta donde guardar los datos con features
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.data.to_csv(output_path, index=False)
            logger.info(f"Datos con features guardados en {output_path}")
            
            # Guardar reporte de features
            report_path = output_path.parent / "feature_report.json"
            with open(report_path, "w") as f:
                json.dump(self.feature_report, f, indent=4)
            logger.info(f"Reporte de features guardado en {report_path}")
            
        except Exception as e:
            logger.error(f"Error al guardar features: {str(e)}")
            raise
            
    def process_pipeline(self) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de feature engineering.
        
        Returns:
            pd.DataFrame: Datos con features
        """
        self.load_data()
        self.create_time_features()
        self.create_rolling_features()
        self.create_categorical_features()
        self.create_interaction_features()
        feature_report = self.generate_feature_report()
        self.save_features()
        
        return self.data

if __name__ == "__main__":
    # Ejemplo de uso
    engineer = FeatureEngineer()
    data_with_features = engineer.process_pipeline()
    
    print("\nReporte de Features:")
    print(json.dumps(engineer.feature_report, indent=2))
    
    print("\nPrimeras filas de datos con features:")
    print(data_with_features.head()) 