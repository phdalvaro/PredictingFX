# Guía de Contribución

## Proceso de Contribución

1. **Fork del Repositorio**
   - Crea un fork del repositorio principal
   - Clona tu fork localmente

2. **Crear Rama de Feature**
   ```bash
   git checkout -b feature/nombre-de-tu-feature
   ```

3. **Desarrollo**
   - Sigue las guías de estilo de código
   - Asegúrate de que el código pase las pruebas
   - Actualiza la documentación según sea necesario

4. **Commit de Cambios**
   - Usa mensajes de commit descriptivos
   - Sigue el formato: "tipo: descripción"
   - Tipos: feat, fix, docs, style, refactor, test, chore

5. **Push a tu Fork**
   ```bash
   git push origin feature/nombre-de-tu-feature
   ```

6. **Pull Request**
   - Crea un PR desde tu rama al main
   - Describe los cambios detalladamente
   - Adjunta capturas de pantalla si es necesario

## Guías de Estilo

### Python
- Sigue PEP 8
- Usa docstrings para todas las funciones
- Máximo 79 caracteres por línea
- 4 espacios para indentación

### Git
- Commits atómicos
- Mensajes descriptivos
- No commits de archivos temporales

### Documentación
- Actualiza README.md si es necesario
- Documenta nuevas features
- Mantén la documentación actualizada

## Estructura del Proyecto

```
ToDo/Fx/
├── data/           # Datos
├── src/            # Código fuente
├── tests/          # Pruebas
├── results/        # Resultados
└── docs/           # Documentación
```

## Pruebas

### Pruebas Unitarias
```bash
python -m pytest tests/unit/
```

### Pruebas de Integración
```bash
python -m pytest tests/integration/
```

## Flujo de Trabajo

1. **Desarrollo Local**
   - Crea un entorno virtual
   - Instala dependencias
   - Ejecuta pruebas

2. **Revisión de Código**
   - Revisa tu propio código
   - Asegúrate de que pasa las pruebas
   - Verifica la documentación

3. **Pull Request**
   - Describe los cambios
   - Adjunta pruebas
   - Espera la revisión

## Contacto

Para preguntas o dudas:
- Email: [EMAIL]
- Issues: [URL_ISSUES]
- Pull Requests: [URL_PRS] 