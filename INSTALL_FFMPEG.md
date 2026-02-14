# Instalación de FFmpeg para Windows

FFmpeg es necesario para que `pydub` pueda procesar archivos de audio en diferentes formatos (como webm, mp3, etc.).

## Opción 1: Instalación con Chocolatey (Recomendado)

Si tienes Chocolatey instalado, ejecuta en PowerShell como administrador:

```powershell
choco install ffmpeg
```

## Opción 2: Instalación Manual

### Paso 1: Descargar FFmpeg

1. Ve a la página oficial: https://www.gyan.dev/ffmpeg/builds/
2. Descarga la versión "ffmpeg-release-essentials.zip"
3. Extrae el archivo ZIP en una ubicación permanente, por ejemplo:
   ```
   C:\ffmpeg
   ```

### Paso 2: Agregar FFmpeg al PATH

1. Abre el Panel de Control
2. Ve a "Sistema y Seguridad" > "Sistema" > "Configuración avanzada del sistema"
3. Haz clic en "Variables de entorno"
4. En "Variables del sistema", busca la variable "Path" y haz clic en "Editar"
5. Haz clic en "Nuevo" y agrega la ruta a la carpeta `bin` de FFmpeg:
   ```
   C:\ffmpeg\bin
   ```
6. Haz clic en "Aceptar" en todas las ventanas

### Paso 3: Verificar la Instalación

Abre una nueva ventana de PowerShell o CMD y ejecuta:

```powershell
ffmpeg -version
```

Deberías ver información sobre la versión de FFmpeg instalada.

## Opción 3: Instalación con Winget

Si tienes Windows 10/11 con winget:

```powershell
winget install ffmpeg
```

## Solución de Problemas

### Error: "Couldn't find ffprobe or avprobe"

Este error significa que FFmpeg no está instalado o no está en el PATH del sistema.

**Solución:**
1. Verifica que FFmpeg esté instalado ejecutando `ffmpeg -version` en una nueva terminal
2. Si no funciona, asegúrate de haber agregado la carpeta `bin` de FFmpeg al PATH
3. **Importante:** Cierra y vuelve a abrir tu terminal/IDE después de modificar el PATH
4. Si estás usando un entorno virtual, desactívalo y vuelve a activarlo

### Verificar que Python puede encontrar FFmpeg

Ejecuta este script de Python para verificar:

```python
import subprocess
try:
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
    print("FFmpeg encontrado:")
    print(result.stdout[:200])
except FileNotFoundError:
    print("ERROR: FFmpeg no está instalado o no está en el PATH")
```

## Reiniciar el Servidor

Después de instalar FFmpeg, **debes reiniciar tu servidor Flask** para que los cambios surtan efecto:

1. Detén el servidor (Ctrl+C)
2. Vuelve a iniciar el servidor:
   ```powershell
   python run_web.py
   ```

## Notas Adicionales

- El warning de pydub desaparecerá una vez que FFmpeg esté correctamente instalado
- FFmpeg es necesario para convertir audio webm (del navegador) a formato WAV
- Sin FFmpeg, el procesamiento de audio fallará con error 400
