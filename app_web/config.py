import os
import sys
from pathlib import Path
import json

# Agregar el directorio padre al path para importar constants
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import WordsConfig

class Config:
    # Configuración de la aplicación
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'una-clave-secreta-muy-segura'
    
    # Rutas
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    WORDS_JSON = os.path.join(MODELS_DIR, 'words.json')
    
    # Configuración del modelo
    MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(MODELS_DIR, 'letters_model.keras'))
    MODEL_METADATA_PATH = os.path.join(MODELS_DIR, 'letters_model_metadata.json')
    
    # Configuración de MediaPipe
    MIN_DETECTION_CONFIDENCE = 0.8
    MIN_TRACKING_CONFIDENCE = 0.8
    
    # Configuración de predicción
    SEQUENCE_LENGTH = 90
    MIN_CONFIDENCE = 0.8
    MIN_CONSECUTIVE_FRAMES = 3
    
    # Clases del modelo
    @property
    def CLASSES(self):
        try:
            with open(self.WORDS_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
            classes = data.get('word_ids', [])
            return [c.lower() for c in classes]
        except Exception:
            return WordsConfig.get_classes()
