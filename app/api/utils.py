import os
import numpy as np
from typing import List, Dict, Any, Union
from werkzeug.utils import secure_filename
from flask import jsonify

# Extensiones de archivo permitidas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename: str) -> bool:
    """Verifica si el archivo tiene una extensión permitida."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file) -> tuple:
    """Valida el archivo de imagen."""
    if not file:
        return False, 'No se proporcionó ningún archivo'
    
    if not allowed_file(file.filename):
        return False, 'Tipo de archivo no permitido'
    
    return True, ''

def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Normaliza los keypoints para que estén en el rango [0, 1]."""
    if len(keypoints) == 0:
        return keypoints
    
    # Asegurarse de que los keypoints estén en 2D
    if keypoints.ndim > 1:
        keypoints = keypoints.reshape(-1)
    
    # Normalizar a [0, 1]
    min_val = np.min(keypoints)
    max_val = np.max(keypoints)
    
    if max_val > min_val:  # Evitar división por cero
        keypoints = (keypoints - min_val) / (max_val - min_val)
    
    return keypoints

def create_response(success: bool, 
                   message: str = '', 
                   data: Union[Dict, List, None] = None,
                   status_code: int = 200) -> tuple:
    """Crea una respuesta JSON estandarizada."""
    response = {
        'success': success,
        'message': message
    }
    
    if data is not None:
        response['data'] = data
    
    return jsonify(response), status_code

def save_uploaded_file(file, upload_folder: str) -> str:
    """Guarda un archivo subido en el servidor."""
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    
    return filepath

def cleanup_old_files(directory: str, max_files: int = 100) -> None:
    """Elimina los archivos más antiguos si se excede el número máximo permitido."""
    if not os.path.exists(directory):
        return
    
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    files = [f for f in files if os.path.isfile(f)]
    
    if len(files) > max_files:
        # Ordenar por fecha de modificación (más antiguo primero)
        files.sort(key=lambda x: os.path.getmtime(x))
        
        # Eliminar los archivos más antiguos
        for file in files[:-max_files]:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error al eliminar archivo {file}: {e}")