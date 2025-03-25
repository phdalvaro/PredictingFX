print("¡Python está funcionando en Cursor! 🚀")
import pandas as pd

# Cargar el archivo Excel
file_path = './data/Copy of FLUCTUA 21-23 (1).xlsx'

# Leer la hoja "OPES"
data = pd.read_excel(file_path, sheet_name='OPES')

# Mostrar las primeras 5 filas
print(data.head())
