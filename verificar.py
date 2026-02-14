import subprocess
try:
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
    print("FFmpeg encontrado:")
    print(result.stdout[:200])
except FileNotFoundError:
    print("ERROR: FFmpeg no está instalado o no está en el PATH")