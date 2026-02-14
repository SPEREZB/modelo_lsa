from flask import Blueprint, render_template, request, jsonify, current_app, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room, send
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
from datetime import datetime
from app_web.core.prediction import LSPPredictor 
import tensorflow as tf 
import io
import base64
from PIL import Image
from functools import lru_cache
from flask import request
from threading import Lock

# Crear el Blueprint principal
main = Blueprint('main', __name__)

# Importar módulo de palabras
from app_web.modules.words import words_blueprint

# Inicializar SocketIO
socketio = SocketIO(cors_allowed_origins="*")

# Control de procesamiento asíncrono por cliente
_processing_clients = set()
_processing_lock = Lock()
_predict_lock = Lock()

# Diccionario para almacenar las secuencias de cada cliente
client_sequences = {}

# Cargar modelo de forma perezosa
@lru_cache(maxsize=1)
def load_cached_model(model_path):
    return load_model(model_path)

# Cargar el modelo
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models/letters_model.keras')

# Inicializar global_predictor como None
global_predictor = None

try:
    # Cargar el modelo
    model = load_cached_model(MODEL_PATH)
    print(f"✅ Modelo cargado correctamente. Tipo: {type(model)}")
    
    # Verificar que el modelo tenga el método predict
    if not hasattr(model, 'predict'):
        raise AttributeError("El modelo no tiene el método 'predict'")
    
    # Mostrar resumen del modelo (si está disponible)
    try:
        model.summary()
    except Exception as summary_error:
        print("⚠️ No se pudo mostrar el resumen del modelo:", summary_error)
    
    # Inicializar el predictor global
    global_predictor = LSPPredictor(model=model, sequence_length=107)
    print("✅ Predictor global inicializado correctamente")
    try:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with _predict_lock:
            global_predictor.process_frame(dummy_frame)
        print("⚙️ Predictor calentado con frame de prueba")
    except Exception as warmup_error:
        print(f"⚠️ Error durante warm-up del predictor: {warmup_error}")

except Exception as e:
    print(f"❌ Error al inicializar el modelo: {str(e)}")
  

 

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

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Maneja la conexión de un nuevo cliente"""
    print('Cliente conectado:', request.sid)
    client_sequences[request.sid] = []
    emit('connection_response', {'data': 'Conectado al servidor de reconocimiento LSP'})

@socketio.on('disconnect')
def handle_disconnect():
    """Maneja la desconexión de un cliente"""
    print('Cliente desconectado:', request.sid)
    if request.sid in client_sequences:
        del client_sequences[request.sid]
    with _processing_lock:
        _processing_clients.discard(request.sid)


@socketio.on('test')
def test():
    """Maneja la desconexión de un cliente"""
    print('BIEN')
    

def _process_frame_async(sid, data):
    try:
        if not isinstance(data, bytes):
            socketio.emit(
                'prediction',
                {'success': False, 'error': 'Datos no binarios recibidos'},
                to=sid,
            )
            return

        # Decodificar imagen de forma optimizada
        nparr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None or image.size == 0:
            socketio.emit(
                'prediction',
                {
                    'success': False,
                    'error': 'No se pudo decodificar la imagen',
                },
                to=sid,
            )
            return

        if global_predictor is None:
            socketio.emit(
                'prediction',
                {'success': False, 'error': 'Modelo no disponible'},
                to=sid,
            )
            return

        # Procesar frame con lock mínimo
        with _predict_lock:
            prediction, confidence = global_predictor.process_frame(image)
            debug = global_predictor.get_debug_status()

        # Emitir resultado fuera del lock para liberar más rápido
        socketio.emit(
            'prediction',
            {
                'success': True,
                'prediction': prediction if prediction else "",
                'confidence': float(confidence) if confidence else 0.0,
                'debug': debug,
            },
            to=sid,
        )

    except Exception as e:
        print(f'Error procesando el frame: {str(e)}')
        socketio.emit(
            'prediction',
            {'success': False, 'error': str(e)},
            to=sid,
        )
    finally:
        with _processing_lock:
            _processing_clients.discard(sid)


@socketio.on('video_frame')
def handle_video_frame(data):
    sid = request.sid

    with _processing_lock:
        if sid in _processing_clients:
            return
        _processing_clients.add(sid)

    socketio.start_background_task(_process_frame_async, sid, data)

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

@main.route('/speech')
def speech():
    return render_template('speech.html')


@main.route('/avatar')
def avatar():
    return render_template('avatar.html', 
                         static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static'))