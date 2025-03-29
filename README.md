# Modelo de Predicción de Volúmenes FX

## Descripción
Sistema de predicción de volúmenes de operaciones FX utilizando machine learning, implementado en producción con monitoreo continuo y validaciones de seguridad.

## Características Principales
- Predicción de volúmenes FX usando XGBoost
- Sistema de monitoreo en tiempo real
- Detección de drift de datos
- Dashboard interactivo
- Validaciones de seguridad robustas
- Sistema de logging detallado
- Pruebas de integración completas

## Estructura del Proyecto
```
Fx/
├── src/
│   ├── models/
│   │   └── production_model.py
│   ├── data/
│   │   ├── data_processor.py
│   │   └── feature_engineering.py
│   └── utils/
│       └── security.py
├── tests/
│   ├── unit/
│   │   └── test_production_model.py
│   └── integration/
│       └── test_model_pipeline.py
├── docs/
│   ├── api.md
│   └── security.md
├── models/
│   ├── xgboost_model.joblib
│   └── scaler.joblib
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   ├── predictions_history.json
│   ├── metrics_history.json
│   └── dashboard.html
├── logs/
│   └── security/
├── security/
│   └── key.key
├── requirements.txt
├── setup.py
├── README.md
├── README2.md
├── CONTRIBUTING.md
└── CHANGELOG.md
```

## Requisitos
- Python 3.8+
- Dependencias listadas en `requirements.txt`

## Instalación
1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/fx-prediction.git
cd fx-prediction
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso
1. Configurar las rutas de los modelos y datos en `config.py`
2. Ejecutar el modelo:
```bash
python src/models/production_model.py
```

## Monitoreo
- Dashboard interactivo en `results/dashboard.html`
- Logs de seguridad en `logs/security/`
- Métricas de rendimiento en `results/metrics_history.json`

## Seguridad
- Encriptación de datos sensibles
- Validación de integridad de modelos
- Rotación automática de logs
- Permisos de archivos seguros

## Pruebas
```bash
# Ejecutar todas las pruebas
pytest

# Ejecutar pruebas unitarias
pytest tests/unit/

# Ejecutar pruebas de integración
pytest tests/integration/
```

## Contribución
Ver [CONTRIBUTING.md](CONTRIBUTING.md) para detalles sobre el proceso de contribución.

## Licencia
Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Changelog
Ver [CHANGELOG.md](CHANGELOG.md) para la lista de cambios.
