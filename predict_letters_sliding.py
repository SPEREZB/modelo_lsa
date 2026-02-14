import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import json
import os
from constants import WordsConfig
import time  # Para manejar el tiempo de las predicciones
from text_to_speech import text_to_speech  # Importar la función de síntesis de voz
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# Configuración de estilos de dibujo
mp_drawing_styles = mp.solutions.drawing_styles

# --- Configuración ---
MODEL_PATH = r"C:\Users\Perez\Desktop\Desarrollo\Modelo_LSP\modelo_lstm_lsp\models\letters_model.keras"

# Preguntar al usuario qué modo de detección desea
print("Seleccione el modo de detección:")
print("1. Solo manos")
print("2. Manos y rostro")
detection_mode = input("Ingrese 1 o 2: ")
include_face = (detection_mode == "2")
print(f"Modo de detección: {'Manos y rostro' if include_face else 'Solo manos'}")

# Configuración de keypoints
NUM_HANDS_KEYPOINTS = 126  # 21 puntos * 3 coordenadas * 2 manos
NUM_FEATURES = NUM_HANDS_KEYPOINTS  # Solo usamos puntos de manos

frame_count = 0
last_pred = None
last_prediction_time = 0
prediction_start_time = {}  # Diccionario para rastrear cuándo comenzó cada predicción
current_stable_pred = None  # Predicción estable actual
STABLE_TIME_THRESHOLD = 5.0  # 5 segundos para considerar una predicción estable

# Configuración de rendimiento mejorada
MIN_CONSECUTIVE_FRAMES = 10  # Aumentado de 3 a 10 para requerir más consistencia
MIN_STABLE_FRAMES = 15       # Mínimo de frames para considerar una predicción estable
CONFIDENCE_THRESHOLD = 0.85  # Aumentado el umbral de confianza
PREDICTION_INTERVAL = 3       # Reducir la frecuencia de predicciones para mejor rendimiento
top_predictions = []
stable_prediction_frames = 0  # Contador de frames estables
# Cargar metadata
with open(MODEL_PATH.replace('.keras', '_metadata.json'), 'r') as f:
    metadata = json.load(f)
    SEQUENCE_LENGTH = metadata['max_sequence_length']  # Ej: 107
    NUM_FEATURES = metadata.get('num_features', NUM_HANDS_KEYPOINTS)  # Usar el valor guardado o el predeterminado (solo manos)

# Diccionario combinado: letras + palabras
LETTERS = WordsConfig.LETTERS
WORDS = WordsConfig.WORDS
CLASSES = WordsConfig.get_classes()

# Verificar que tenemos 45 clases (27 letras + 18 palabras)
print(f"Número de clases: {len(CLASSES)}")
print("Clases ordenadas:")
for i, clase in enumerate(CLASSES):
    print(f"{i}: {clase}")

# Cargar modelo
model = load_model(MODEL_PATH)

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Inicializar detección de manos
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inicializar detección de rostro si es necesario
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) if include_face else None

def get_top_predictions(predictions, classes, top_n=3):
    """Obtiene las top N predicciones con sus porcentajes de confianza"""
    # Obtener los índices de las predicciones más altas
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    # Crear lista de tuplas (clase, confianza)
    top_predictions = [(classes[i], predictions[i] * 100) for i in top_indices 
                      if i < len(classes) and predictions[i] > 0.01]  # Filtrar predicciones muy bajas
    return top_predictions

# Función para extraer keypoints
def extract_keypoints(frame):
    # Convertir a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar manos
    hand_results = hands.process(image_rgb)
    keypoints = []

    # Extraer keypoints de manos
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
    
    # Extraer keypoints de la cara si está habilitado
    if include_face and face_mesh:
        face_results = face_mesh.process(image_rgb)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for lm in face_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
    
    # Asegurar que tenemos el número correcto de características
    if len(keypoints) < NUM_FEATURES:
        keypoints.extend([0.0] * (NUM_FEATURES - len(keypoints)))
    elif len(keypoints) > NUM_FEATURES:
        keypoints = keypoints[:NUM_FEATURES]
    
    # Dibujar landmarks si se está mostrando el video
    if include_face and face_mesh and hasattr(face_results, 'multi_face_landmarks') and face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                face_landmarks, 
                mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                .get_default_face_mesh_tesselation_style()
            )
    
    return np.array(keypoints, dtype=np.float32), hand_results

# --- Captura de cámara ---
cap = cv2.VideoCapture(0)
sequence = []
prediction = ""
last_spoken = ""  

# Sistema de votación para estabilizar predicciones
prediction_history = []
MIN_CONSECUTIVE_FRAMES = 5  
CONFIDENCE_THRESHOLD = 0.3   
last_top_pred = None  

print("Presiona 'q' para salir.")

 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    keypoints, results = extract_keypoints(frame)
    
    sequence.append(keypoints)
    # Mantener solo los últimos SEQUENCE_LENGTH frames
    sequence = sequence[-SEQUENCE_LENGTH:]

    if len(sequence) < SEQUENCE_LENGTH:
        # Padding con ceros para completar la secuencia
        pad = [np.zeros(NUM_FEATURES, dtype=np.float32) for _ in range(SEQUENCE_LENGTH - len(sequence))]
        sequence_padded = sequence + pad
    else:
        sequence_padded = sequence[-SEQUENCE_LENGTH:]

    # Convertir a array 3D: (1, SEQUENCE_LENGTH, NUM_FEATURES)
    seq_array = np.array(sequence_padded, dtype=np.float32)
    seq_array = np.expand_dims(seq_array, axis=0)

    # Predicción (solo procesar cada PREDICTION_INTERVAL frames)
    frame_count += 1

    if frame_count % PREDICTION_INTERVAL == 0:
        pred = model.predict(seq_array, verbose=0)[0]  # Obtener el array de predicciones
        last_pred = pred
        # Obtener las 3 mejores predicciones
        top_predictions = get_top_predictions(pred, CLASSES, 3)
     
        for i, (cls, conf) in enumerate(top_predictions, 1):
            print(f"{i}. {cls}: {conf:.1f}%")
    else:
        pred = last_pred if last_pred is not None else np.zeros(len(CLASSES))
    pred_idx = np.argmax(pred)
    
    # Obtener la predicción principal
    current_top_pred = CLASSES[pred_idx] if 0 <= pred_idx < len(CLASSES) else None
    current_time = time.time()
    
    # Si la predicción principal ha cambiado, reiniciar el contador
    if current_top_pred != last_top_pred:
        prediction_history = []  # Reiniciar el historial ante un cambio de predicción
        current_stable_pred = None
    last_top_pred = current_top_pred
    
    # Actualizar el tiempo de inicio para la predicción actual
    if current_top_pred is not None:
        if current_top_pred not in prediction_start_time:
            prediction_start_time[current_top_pred] = current_time
    
    # Verificar si alguna predicción ha estado presente por más de 5 segundos
    stable_prediction = None
    for pred_name, start_time in list(prediction_start_time.items()):
        if current_time - start_time >= STABLE_TIME_THRESHOLD:
            stable_prediction = pred_name
            break
    
    # Sistema de predicción mejorado
    if 0 <= pred_idx < len(CLASSES) and pred[pred_idx] >= CONFIDENCE_THRESHOLD:
        current_pred = CLASSES[pred_idx]
        
        # Si es la misma predicción que la anterior, incrementar el contador
        if current_pred == last_top_pred:
            current_stable_frames += 1
        else:
            current_stable_frames = 1
            
        last_top_pred = current_pred
        
        # Solo considerar la predicción si ha sido estable por suficientes frames
        if current_stable_frames >= MIN_STABLE_FRAMES:
            # Agregar predicción al historial
            prediction_history.append((current_pred, pred[pred_idx]))
            
            # Mantener un historial razonable
            if len(prediction_history) > MIN_CONSECUTIVE_FRAMES:
                prediction_history.pop(0)
                
                # Calcular la confianza promedio de la predicción
                pred_confidence = sum(conf for p, conf in prediction_history if p == current_pred)
                avg_confidence = pred_confidence / MIN_CONSECUTIVE_FRAMES
                
                # Solo aceptar si la confianza promedio es alta
                if avg_confidence >= CONFIDENCE_THRESHOLD * 1.2:  # Umbral más estricto para la media
                    prediction = current_pred
                    # Si hay una predicción estable por suficiente tiempo, reiniciar contadores
                    if stable_prediction is not None and current_pred == stable_prediction:
                        prediction_history = []
                        current_stable_frames = 0
                        prediction_start_time = {}
    else:
        # Si la confianza es baja, reiniciar contadores
        current_stable_frames = 0
        last_top_pred = None
        prediction = ""  # No hacer predicción si la confianza es baja
    
    # Limpiar la predicción si no hay manos
    if not results.multi_hand_landmarks:
        prediction = ""
        prediction_history = []  # Limpiar historial cuando no hay manos
        prediction_start_time = {}  # Limpiar los tiempos de predicción
    # Dibujar landmarks de manos
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),  # Rojo
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # Rojo
            )

        
        # Mostrar información de análisis en tiempo real
        y_offset = 30
        
        # Barra de progreso de estabilidad
        stability_pct = min(100, (current_stable_frames / MIN_STABLE_FRAMES) * 100)
        cv2.rectangle(frame, (10, y_offset), (210, y_offset + 15), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, y_offset), (10 + int(2 * stability_pct), y_offset + 15), 
                     (0, int(255 * (stability_pct/100)), int(255 * (1 - stability_pct/200))), -1)
        cv2.putText(frame, f"Estabilidad: {int(stability_pct)}%", (15, y_offset + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 30
        
        # Título de las predicciones
        cv2.putText(frame, "Predicciones:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 25
        
        # Mostrar las 3 mejores predicciones
        for i, (cls, conf) in enumerate(top_predictions, 1):
            text = f"{i}. {cls.upper()}: {conf:.1f}%"
            is_high_confidence = conf >= CONFIDENCE_THRESHOLD * 100
            is_stable = current_stable_frames >= MIN_STABLE_FRAMES and cls == last_top_pred
            
            if is_stable and is_high_confidence:
                color = (0, 255, 0)  # Verde para predicción estable y confiable
                thickness = 1
                # Fondo para mejor legibilidad
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(frame, (10, y_offset - 20), (20 + text_size[0], y_offset + 5), 
                            (0, 0, 0), -1)
            elif is_high_confidence:
                color = (0, 255, 255)  # Amarillo para alta confianza pero inestable
                thickness = 1
            else:
                color = (200, 200, 200)  # Gris para baja confianza
                thickness = 1
                
            cv2.putText(frame, text, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness, cv2.LINE_AA)
            y_offset += 25
        
        # Mostrar la predicción final cuando sea estable
        if prediction and current_stable_frames >= MIN_STABLE_FRAMES:
            # Fondo semitransparente para mejor legibilidad
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame.shape[0] - 50), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            alpha = 0.7  # Factor de transparencia
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Texto de predicción
            text = f"SEÑAL DETECTADA: {prediction.upper()}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2  # Centrar texto
            cv2.putText(frame, text, (text_x, frame.shape[0] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Hablar la predicción solo si es diferente a la última hablada
        if prediction and prediction != last_spoken:
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if (current_time - last_prediction_time) > 1.0:  # Mínimo 1 segundo entre predicciones
                print(f"Predicción detectada: {prediction}")
                text_to_speech(prediction)
                last_spoken = prediction
                last_prediction_time = current_time


    cv2.imshow("Reconocimiento LSA", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
