import os
from pathlib import Path

class Config:
    # Configuración de la aplicación
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'una-clave-secreta-muy-segura'
    
    # Rutas
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    # Configuración del modelo
    MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(MODELS_DIR, 'letters_model.keras'))
    MODEL_METADATA_PATH = os.path.join(MODELS_DIR, 'letters_model_metadata.json')
    
    # Configuración de MediaPipe
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Configuración de predicción
    SEQUENCE_LENGTH = 90
    MIN_CONFIDENCE = 0.2
    MIN_CONSECUTIVE_FRAMES = 3
    
    # Clases del modelo
    @property
    def CLASSES(self):
        letters = list("abcdefjklmnopqrsuvxz")
        words = [
            "adios", "alumno", "bien", "chau", "comer", "comoestas", 
            "dormir", "el", "gracias", "hola", "informe", "investigar", "leer", 
            "legusta", "mellamo", "nolegusta", "perdon", "tienesrazon", "timido", "yo"
        ]
        return sorted(letters + words, key=lambda x: x.lower())
