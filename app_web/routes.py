from flask import Blueprint, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
from datetime import datetime
from app_web.core.prediction import LSPPredictor 
import tensorflow as tf
from app_web.core.detection import HandDetector

# Importar módulo de palabras
from app_web.modules.words import words_blueprint

# Crear carpeta para guardar imágenes si no existe
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'received_images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  

main = Blueprint('main', __name__)

# Registrar blueprints
main.register_blueprint(words_blueprint)

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)

# Inicializar detector de manos
hand_detector = HandDetector(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7
)
 

# Cargar modelo 
MODEL_PATH = 'models/letters_model.keras'
model = load_model(MODEL_PATH)

# Usar 107 frames para coincidir con el modelo entrenado
predictor = LSPPredictor(model=model, sequence_length=107)
@main.route('/')
def index():
    return render_template('index.html')

@main.route('/api/detect_keypoints', methods=['POST'])
def detect_keypoints():
    if 'image' not in request.files:
        return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    try:
        # Leer la imagen
        nparr = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detectar keypoints
        hand_results = hand_detector.detect(frame)
        
        # Procesar resultados
        keypoints = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
        
        return jsonify({
            'success': True,
            'keypoints': keypoints,
            'num_hands': len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500    

@main.route('/api/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
        
        word = request.form.get('word', '').strip().lower()
        if not word:
            return jsonify({'error': 'No se proporcionó ninguna palabra'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
            
        # Crear directorio para la palabra si no existe
        word_dir = os.path.join(UPLOAD_FOLDER, word)
        os.makedirs(word_dir, exist_ok=True)
        
        # Guardar la imagen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(word_dir, filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Imagen guardada para la palabra: {word}',
            'path': filepath
        })
     
    return render_template('capture.html')

@main.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint para predecir el gesto de la lengua de señas."""
    if 'image' not in request.files:
        return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
    
    try:
        # Leer la imagen
        print("Longitud de files: ", len(request.files))
        file = request.files['image']
        
        # Guardar la imagen recibida
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Leer la imagen para procesamiento
        nparr = np.frombuffer(open(filepath, 'rb').read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Procesar el frame
        prediction, confidence = predictor.process_frame(image)
        print(prediction)
        return jsonify({
            'success': True,
            'prediction': prediction 
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500