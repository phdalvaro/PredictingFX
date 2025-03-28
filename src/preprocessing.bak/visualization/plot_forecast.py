import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/plot_forecast.log'),
        logging.StreamHandler()
    ]
)

def load_forecast_data(base_path):
    """Cargar datos de pronóstico"""
    try:
        forecasts_dir = os.path.join(base_path, 'data', 'processed', 'forecasts')
        
        # Cargar pronósticos
        volume_forecast = pd.read_csv(os.path.join(forecasts_dir, 'volume_forecast.csv'))
        transaction_forecast = pd.read_csv(os.path.join(forecasts_dir, 'transaction_forecast.csv'))
        
        # Convertir columnas de fecha
        volume_forecast['Fecha'] = pd.to_datetime(volume_forecast['Fecha'])
        transaction_forecast['Fecha'] = pd.to_datetime(transaction_forecast['Fecha'])
        
        logging.info("Datos de pronóstico cargados exitosamente")
        return volume_forecast, transaction_forecast
    except Exception as e:
        logging.error(f"Error al cargar datos de pronóstico: {str(e)}")
        raise

def create_volume_plot(volume_forecast):
    """Crear gráfico de pronóstico de volumen"""
    try:
        fig = go.Figure()
        
        # Agregar línea de pronóstico
        fig.add_trace(go.Scatter(
            x=volume_forecast['Fecha'],
            y=volume_forecast['Volumen_Pronosticado'],
            mode='lines+markers',
            name='Pronóstico de Volumen',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        # Actualizar layout
        fig.update_layout(
            title='Pronóstico de Volumen de Operaciones FX',
            xaxis_title='Fecha',
            yaxis_title='Volumen (USD)',
            template='plotly_white',
            showlegend=True,
            height=600
        )
        
        # Agregar rangos de fecha
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1 semana", step="day", stepmode="backward"),
                    dict(count=14, label="2 semanas", step="day", stepmode="backward"),
                    dict(count=30, label="1 mes", step="day", stepmode="backward"),
                    dict(step="all", label="Todo")
                ])
            )
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error al crear gráfico de volumen: {str(e)}")
        raise

def create_transaction_plot(transaction_forecast):
    """Crear gráfico de pronóstico de transacciones"""
    try:
        fig = go.Figure()
        
        # Colores para diferentes horizontes
        colors = {
            '7d': '#1f77b4',
            '14d': '#ff7f0e',
            '30d': '#2ca02c'
        }
        
        # Agregar líneas para cada horizonte
        for horizon in ['7d', '14d', '30d']:
            col_name = f'Probabilidad_Transaccion_{horizon}'
            fig.add_trace(go.Scatter(
                x=transaction_forecast['Fecha'],
                y=transaction_forecast[col_name],
                mode='lines+markers',
                name=f'Probabilidad {horizon}',
                line=dict(color=colors[horizon], width=2),
                marker=dict(size=6)
            ))
        
        # Actualizar layout
        fig.update_layout(
            title='Probabilidad de Próximas Transacciones',
            xaxis_title='Fecha',
            yaxis_title='Probabilidad',
            template='plotly_white',
            showlegend=True,
            height=600
        )
        
        # Configurar eje Y para mostrar probabilidades
        fig.update_yaxes(range=[0, 1], tickformat='.2%')
        
        # Agregar rangos de fecha
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1 semana", step="day", stepmode="backward"),
                    dict(count=14, label="2 semanas", step="day", stepmode="backward"),
                    dict(count=30, label="1 mes", step="day", stepmode="backward"),
                    dict(step="all", label="Todo")
                ])
            )
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error al crear gráfico de transacciones: {str(e)}")
        raise

def save_plots(volume_plot, transaction_plot, base_path):
    """Guardar gráficos"""
    try:
        visualizations_dir = os.path.join(base_path, 'data', 'processed', 'visualizations')
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # Guardar gráficos
        volume_plot.write_html(os.path.join(visualizations_dir, 'volume_forecast.html'))
        transaction_plot.write_html(os.path.join(visualizations_dir, 'transaction_forecast.html'))
        
        logging.info("Gráficos guardados exitosamente")
    except Exception as e:
        logging.error(f"Error al guardar gráficos: {str(e)}")
        raise

def main():
    try:
        # Obtener ruta base del proyecto
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Crear directorio de logs si no existe
        os.makedirs('logs', exist_ok=True)
        
        logging.info("Iniciando generación de visualizaciones...")
        
        # 1. Cargar datos de pronóstico
        volume_forecast, transaction_forecast = load_forecast_data(base_path)
        
        # 2. Crear gráficos
        volume_plot = create_volume_plot(volume_forecast)
        transaction_plot = create_transaction_plot(transaction_forecast)
        
        # 3. Guardar gráficos
        save_plots(volume_plot, transaction_plot, base_path)
        
        logging.info("Proceso de generación de visualizaciones completado exitosamente")
        
    except Exception as e:
        logging.error(f"Error en el proceso de generación de visualizaciones: {str(e)}")
        raise

if __name__ == "__main__":
    main() 