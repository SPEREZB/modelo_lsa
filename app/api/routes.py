from flask import request, jsonify
import cv2
import numpy as np
from ..services.mediapipe_service import MediaPipeService
from ..core.prediction import LSPPredictor
from .. import config
import tensorflow as tf
from . import api_bp

# Inicializar servicios
mediapipe_service = MediaPipeService()

# Cargar el modelo y configurar el predictor
model = tf.keras.models.load_model(config.Config.MODEL_PATH)
# Usar 107 frames para coincidir con el modelo entrenado
predictor = LSPPredictor(model=model, sequence_length=107)

@api_bp.route('/detect', methods=['POST'])
def detect():
    """Endpoint para detectar puntos clave en una imagen."""
    if 'image' not in request.files:
        return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
    
    try:
        # Leer la imagen
        file = request.files['image']
        nparr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detectar puntos clave
        results = mediapipe_service.detect(image)
        
        # Dibujar landmarks (opcional)
        if request.args.get('draw', 'false').lower() == 'true':
            image = mediapipe_service.draw_landmarks(image, results)
            _, buffer = cv2.imencode('.jpg', image)
            return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}
        
        # Extraer keypoints
        include_face = request.args.get('include_face', 'false').lower() == 'true'
        keypoints = mediapipe_service.extract_keypoints(results, include_face)
        
        return jsonify({
            'success': True,
            'keypoints': keypoints.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/predict', methods=['POST'])
def predict():
    """Endpoint para predecir el gesto de la lengua de señas."""
    if 'image' not in request.files:
        return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
    
    try:
        # Leer la imagen
        file = request.files['image']
        nparr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Procesar el frame
        prediction, confidence = predictor.process_frame(image)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/reset', methods=['POST'])
def reset():
    """Reinicia el estado del predictor."""
    predictor.reset()
    return jsonify({'success': True})