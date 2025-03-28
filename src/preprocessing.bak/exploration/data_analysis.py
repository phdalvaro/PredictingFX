import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_analysis.log'),
        logging.StreamHandler()
    ]
)

def load_cleaned_data(base_path):
    """Cargar datos limpios"""
    try:
        file_path = os.path.join(base_path, 'data', 'processed', 'cleaned_data.csv')
        df = pd.read_csv(file_path)
        df['Fecha de cierre'] = pd.to_datetime(df['Fecha de cierre'])
        logging.info(f"Datos limpios cargados exitosamente: {len(df)} filas")
        return df
    except Exception as e:
        logging.error(f"Error al cargar datos limpios: {str(e)}")
        raise

def analyze_data(df):
    """Realizar análisis básico de los datos"""
    try:
        # Información básica
        info = {
            'Número de filas': len(df),
            'Número de columnas': len(df.columns),
            'Tipos de datos': df.dtypes.to_dict(),
            'Valores faltantes': df.isnull().sum().to_dict(),
            'Estadísticas descriptivas': df.describe().to_dict()
        }
        
        # Análisis temporal
        info['Rango de fechas'] = {
            'Inicio': df['Fecha de cierre'].min().strftime('%Y-%m-%d'),
            'Fin': df['Fecha de cierre'].max().strftime('%Y-%m-%d')
        }
        
        # Análisis por cliente
        info['Top 10 clientes por volumen'] = df.groupby('Codigo del Cliente (IBS)')['Importe FX'].sum().nlargest(10).to_dict()
        
        # Análisis por producto
        info['Distribución por producto'] = df['Producto'].value_counts().to_dict()
        
        # Análisis por tipo de operación
        info['Distribución por tipo de operación'] = df['Tipo de Operación'].value_counts().to_dict()
        
        logging.info("Análisis básico completado exitosamente")
        return info
    except Exception as e:
        logging.error(f"Error en análisis básico: {str(e)}")
        raise

def create_visualizations(df, output_path):
    """Crear visualizaciones"""
    try:
        # Crear directorio si no existe
        os.makedirs(output_path, exist_ok=True)
        
        # 1. Evolución temporal del volumen
        fig_volume = px.line(df, x='Fecha de cierre', y='Importe FX',
                           title='Evolución del Volumen de Transacciones',
                           labels={'Importe FX': 'Volumen (USD)', 'Fecha de cierre': 'Fecha'})
        fig_volume.write_html(os.path.join(output_path, 'volume_trend.html'))
        
        # 2. Distribución de importes
        fig_dist = px.histogram(df, x='Importe FX',
                              title='Distribución de Importes',
                              labels={'Importe FX': 'Volumen (USD)', 'count': 'Frecuencia'})
        fig_dist.write_html(os.path.join(output_path, 'amount_distribution.html'))
        
        # 3. Top 10 clientes por volumen
        top_clients = df.groupby('Codigo del Cliente (IBS)')['Importe FX'].sum().nlargest(10)
        fig_clients = px.bar(x=top_clients.index, y=top_clients.values,
                           title='Top 10 Clientes por Volumen',
                           labels={'x': 'Código de Cliente', 'y': 'Volumen Total (USD)'})
        fig_clients.write_html(os.path.join(output_path, 'top_clients.html'))
        
        # 4. Análisis de tipo de cambio
        fig_tc = make_subplots(rows=2, cols=1,
                             subplot_titles=('Evolución del Tipo de Cambio', 'Distribución del Tipo de Cambio'))
        
        fig_tc.add_trace(
            go.Scatter(x=df['Fecha de cierre'], y=df['T/C Cerrado'], name='T/C Cerrado'),
            row=1, col=1
        )
        
        fig_tc.add_trace(
            go.Histogram(x=df['T/C Cerrado'], name='Distribución T/C'),
            row=2, col=1
        )
        
        fig_tc.update_layout(height=800, title_text="Análisis del Tipo de Cambio")
        fig_tc.write_html(os.path.join(output_path, 'exchange_rate_analysis.html'))
        
        # 5. Análisis por producto y tipo de operación
        fig_product = px.pie(df, names='Producto', title='Distribución por Producto')
        fig_product.write_html(os.path.join(output_path, 'product_distribution.html'))
        
        fig_operation = px.pie(df, names='Tipo de Operación', title='Distribución por Tipo de Operación')
        fig_operation.write_html(os.path.join(output_path, 'operation_distribution.html'))
        
        logging.info(f"Visualizaciones guardadas en: {output_path}")
    except Exception as e:
        logging.error(f"Error al crear visualizaciones: {str(e)}")
        raise

def generate_report(info, output_path):
    """Generar reporte HTML"""
    try:
        # Obtener las columnas numéricas disponibles
        numeric_cols = ['Importe FX', 'T/C Cerrado', 'PX_MID']
        available_cols = [col for col in numeric_cols if col in info['Estadísticas descriptivas']]
        
        html_content = f"""
        <html>
        <head>
            <title>Análisis Exploratorio de Datos FX</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .section {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Análisis Exploratorio de Datos FX</h1>
            
            <div class="section">
                <h2>Información General</h2>
                <table>
                    <tr>
                        <th>Métrica</th>
                        <th>Valor</th>
                    </tr>
                    <tr>
                        <td>Número de filas</td>
                        <td>{info['Número de filas']}</td>
                    </tr>
                    <tr>
                        <td>Número de columnas</td>
                        <td>{info['Número de columnas']}</td>
                    </tr>
                    <tr>
                        <td>Rango de fechas</td>
                        <td>{info['Rango de fechas']['Inicio']} - {info['Rango de fechas']['Fin']}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Tipos de Datos</h2>
                <table>
                    <tr>
                        <th>Columna</th>
                        <th>Tipo de Dato</th>
                    </tr>
                    {''.join(f"<tr><td>{col}</td><td>{dtype}</td></tr>" for col, dtype in info['Tipos de datos'].items())}
                </table>
            </div>
            
            <div class="section">
                <h2>Valores Faltantes</h2>
                <table>
                    <tr>
                        <th>Columna</th>
                        <th>Valores Faltantes</th>
                    </tr>
                    {''.join(f"<tr><td>{col}</td><td>{count}</td></tr>" for col, count in info['Valores faltantes'].items() if count > 0)}
                </table>
            </div>
            
            <div class="section">
                <h2>Top 10 Clientes por Volumen</h2>
                <table>
                    <tr>
                        <th>Código de Cliente</th>
                        <th>Volumen Total (USD)</th>
                    </tr>
                    {''.join(f"<tr><td>{client}</td><td>{amount:,.2f}</td></tr>" for client, amount in info['Top 10 clientes por volumen'].items())}
                </table>
            </div>
            
            <div class="section">
                <h2>Distribución por Producto</h2>
                <table>
                    <tr>
                        <th>Producto</th>
                        <th>Frecuencia</th>
                    </tr>
                    {''.join(f"<tr><td>{product}</td><td>{count}</td></tr>" for product, count in info['Distribución por producto'].items())}
                </table>
            </div>
            
            <div class="section">
                <h2>Distribución por Tipo de Operación</h2>
                <table>
                    <tr>
                        <th>Tipo de Operación</th>
                        <th>Frecuencia</th>
                    </tr>
                    {''.join(f"<tr><td>{operation}</td><td>{count}</td></tr>" for operation, count in info['Distribución por tipo de operación'].items())}
                </table>
            </div>
            
            <div class="section">
                <h2>Estadísticas Descriptivas</h2>
                <table>
                    <tr>
                        <th>Métrica</th>
                        {''.join(f"<th>{col}</th>" for col in available_cols)}
                    </tr>
                    {''.join(f"<tr><td>{metric}</td>{''.join(f'<td>{info['Estadísticas descriptivas'][col][metric]:,.2f}</td>' for col in available_cols)}</tr>" for metric in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])}
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizaciones</h2>
                <p>Las siguientes visualizaciones interactivas están disponibles:</p>
                <ul>
                    <li><a href="volume_trend.html">Evolución del Volumen de Transacciones</a></li>
                    <li><a href="amount_distribution.html">Distribución de Importes</a></li>
                    <li><a href="top_clients.html">Top 10 Clientes por Volumen</a></li>
                    <li><a href="exchange_rate_analysis.html">Análisis del Tipo de Cambio</a></li>
                    <li><a href="product_distribution.html">Distribución por Producto</a></li>
                    <li><a href="operation_distribution.html">Distribución por Tipo de Operación</a></li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        report_path = os.path.join(output_path, 'data_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info(f"Reporte HTML generado en: {report_path}")
    except Exception as e:
        logging.error(f"Error al generar reporte HTML: {str(e)}")
        raise

def main():
    try:
        # Obtener ruta base del proyecto
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Crear directorio de logs si no existe
        os.makedirs('logs', exist_ok=True)
        
        # Cargar datos limpios
        logging.info("Iniciando análisis exploratorio...")
        df = load_cleaned_data(base_path)
        
        # Realizar análisis
        info = analyze_data(df)
        
        # Crear directorio para visualizaciones
        output_path = os.path.join(base_path, 'data', 'processed', 'exploration')
        os.makedirs(output_path, exist_ok=True)
        
        # Crear visualizaciones
        create_visualizations(df, output_path)
        
        # Generar reporte
        generate_report(info, output_path)
        
        logging.info("Análisis exploratorio completado exitosamente")
        
    except Exception as e:
        logging.error(f"Error en el análisis exploratorio: {str(e)}")
        raise

if __name__ == "__main__":
    main() 