import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import json
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureSelector:
    def __init__(self, data_path: str = "data/processed/features.csv"):
        """
        Inicializa el selector de features.
        
        Args:
            data_path: Ruta al archivo de datos con features
        """
        self.data_path = Path(data_path)
        self.data = None
        self.X = None
        self.y = None
        self.selected_features = []
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_selector = None
        self.selection_report = {}
        
    def load_data(self) -> None:
        """
        Carga los datos con features.
        """
        try:
            logger.info(f"Cargando datos con features desde {self.data_path}")
            self.data = pd.read_csv(self.data_path, low_memory=False)
            logger.info(f"Datos cargados exitosamente. Shape: {self.data.shape}")
            
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            raise
            
    def prepare_target(self) -> None:
        """
        Prepara la variable objetivo (Importe FX).
        """
        try:
            logger.info("Preparando variable objetivo")
            self.y = self.data['Importe FX'].values
            logger.info(f"Variable objetivo preparada. Shape: {self.y.shape}")
            
        except Exception as e:
            logger.error(f"Error al preparar variable objetivo: {str(e)}")
            raise
            
    def prepare_features(self) -> None:
        """
        Prepara las features para el modelo.
        """
        try:
            logger.info("Preparando features para el modelo")
            
            # Excluir columnas no deseadas
            exclude_cols = [
                'Fecha de cierre',
                'Importe FX',
                'Codigo del Cliente (IBS)',
                'Dia_Nombre'  # Excluir nombres de días por ser redundantes
            ]
            
            # Seleccionar features numéricas
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            self.X = self.data[feature_cols].values
            logger.info(f"Features preparadas. Shape: {self.X.shape}")
            
        except Exception as e:
            logger.error(f"Error al preparar features: {str(e)}")
            raise
            
    def handle_missing_values(self) -> None:
        """
        Maneja valores faltantes en las features.
        """
        try:
            logger.info("Manejando valores faltantes")
            self.X = self.imputer.fit_transform(self.X)
            logger.info("Valores faltantes manejados exitosamente")
            
        except Exception as e:
            logger.error(f"Error al manejar valores faltantes: {str(e)}")
            raise
            
    def scale_features(self) -> None:
        """
        Escala las features usando StandardScaler.
        """
        try:
            logger.info("Escalando features")
            self.X = self.scaler.fit_transform(self.X)
            logger.info("Features escaladas exitosamente")
            
        except Exception as e:
            logger.error(f"Error al escalar features: {str(e)}")
            raise
            
    def select_features(self, k: int = 20) -> None:
        """
        Selecciona las k mejores features usando SelectKBest.
        
        Args:
            k: Número de features a seleccionar
        """
        try:
            logger.info(f"Seleccionando {k} mejores features")
            
            # Obtener las columnas numéricas antes de la transformación
            numeric_cols = [col for col in self.data.select_dtypes(include=[np.number]).columns 
                          if col not in ['Fecha de cierre', 'Importe FX', 'Codigo del Cliente (IBS)']]
            
            logger.info(f"Número de columnas numéricas: {len(numeric_cols)}")
            logger.info(f"Shape de X: {self.X.shape}")
            
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            self.X = self.feature_selector.fit_transform(self.X, self.y)
            
            logger.info(f"Shape de X después de la selección: {self.X.shape}")
            logger.info(f"Longitud de scores: {len(self.feature_selector.scores_)}")
            
            # Obtener scores y crear DataFrame con las features originales
            feature_scores = pd.DataFrame({
                'Feature': numeric_cols[:len(self.feature_selector.scores_)],  # Asegurar misma longitud
                'Score': self.feature_selector.scores_
            })
            
            # Ordenar por score y seleccionar las k mejores
            feature_scores = feature_scores.sort_values('Score', ascending=False)
            self.selected_features = feature_scores.head(k)['Feature'].tolist()
            
            logger.info(f"Features seleccionadas: {self.selected_features}")
            
        except Exception as e:
            logger.error(f"Error al seleccionar features: {str(e)}")
            raise
            
    def split_data(self, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        
        Args:
            test_size: Proporción de datos para prueba
            
        Returns:
            Tuple con X_train, X_test, y_train, y_test
        """
        try:
            logger.info("Dividiendo datos en conjuntos de entrenamiento y prueba")
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42
            )
            logger.info(f"Conjuntos creados. Shapes: X_train: {X_train.shape}, X_test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error al dividir datos: {str(e)}")
            raise
            
    def generate_selection_report(self) -> Dict:
        """
        Genera un reporte de la selección de features.
        """
        selection_report = {
            "total_features_original": self.X.shape[1],
            "selected_features": self.selected_features,
            "feature_importance": {
                feature: float(score) for feature, score in zip(
                    self.data.select_dtypes(include=[np.number]).columns,
                    self.feature_selector.scores_
                )
            }
        }
        
        self.selection_report = selection_report
        return selection_report
        
    def save_selected_data(self, output_dir: str = "data/processed") -> None:
        """
        Guarda los datos seleccionados y el reporte.
        
        Args:
            output_dir: Directorio donde guardar los datos
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar datos seleccionados
            selected_data = pd.DataFrame(
                self.X,
                columns=self.selected_features
            )
            selected_data['Importe_FX'] = self.y
            
            selected_data.to_csv(output_dir / "selected_features.csv", index=False)
            logger.info(f"Datos seleccionados guardados en {output_dir}/selected_features.csv")
            
            # Guardar reporte
            with open(output_dir / "feature_selection_report.json", "w") as f:
                json.dump(self.selection_report, f, indent=4)
            logger.info(f"Reporte de selección guardado en {output_dir}/feature_selection_report.json")
            
        except Exception as e:
            logger.error(f"Error al guardar datos seleccionados: {str(e)}")
            raise
            
    def process_pipeline(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Ejecuta el pipeline completo de selección de features.
        
        Returns:
            Tuple con X_train, X_test, y_train, y_test
        """
        self.load_data()
        self.prepare_target()
        self.prepare_features()
        self.handle_missing_values()
        self.scale_features()
        self.select_features()
        X_train, X_test, y_train, y_test = self.split_data()
        self.generate_selection_report()
        self.save_selected_data()
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Ejemplo de uso
    selector = FeatureSelector()
    X_train, X_test, y_train, y_test = selector.process_pipeline()
    
    print("\nReporte de Selección de Features:")
    print(json.dumps(selector.selection_report, indent=2))
    
    print("\nShapes de los conjuntos de datos:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}") 