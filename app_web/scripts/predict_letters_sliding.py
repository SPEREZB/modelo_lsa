import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import json
import os
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
NUM_HANDS_KEYPOINTS = 126  
NUM_FACE_KEYPOINTS = 1404  

frame_count = 0
last_pred = None
last_prediction_time = 0

# Configuración de rendimiento
MIN_CONSECUTIVE_FRAMES = 3  
CONFIDENCE_THRESHOLD = 0.8
PREDICTION_INTERVAL = 2  
top_predictions = []
# Cargar metadata
with open(MODEL_PATH.replace('.keras', '_metadata.json'), 'r') as f:
    metadata = json.load(f)
    SEQUENCE_LENGTH = metadata['max_sequence_length']  # Ej: 107
    NUM_FEATURES = metadata['num_features'] if 'num_features' in metadata else 1530  # Debe coincidir con el modelo

# Diccionario combinado: letras + palabras
LETTERS = list("abcdefjklmnopqrsuvxz")  # Solo las letras que están en words.json
WORDS = [
    "adios", "alumno", "aprender", "bien", "chau", "cocinar", "comer", "comoestas", 
    "dormir", "el", "estudiar", "gracias", "hola", "informe", "investigar", "leer", 
    "legusta", "mellamo", "nolegusta","perder", "perdon", "tienesrazon", "timido", "yo"
]

# Convertir letras individuales a palabras de un solo carácter
letters_as_words = [letra for letra in LETTERS]

# Combinar letras y palabras y ordenarlas alfabéticamente
CLASSES = sorted(letters_as_words + WORDS, key=lambda x: x.lower())

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
    if current_top_pred != last_top_pred:
        prediction_history = []  # Reiniciar el historial ante un cambio de predicción
    last_top_pred = current_top_pred
    
    # Verificar que el índice esté dentro del rango y cumpla con el umbral de confianza
    if 0 <= pred_idx < len(CLASSES) and pred[pred_idx] >= CONFIDENCE_THRESHOLD:
        current_pred = CLASSES[pred_idx]
        
        # Agregar predicción al historial solo si supera el umbral
        prediction_history.append((current_pred, pred[pred_idx]))  # Guardar predicción y confianza
        
        # Mantener solo las últimas N predicciones
        if len(prediction_history) > MIN_CONSECUTIVE_FRAMES:
            prediction_history.pop(0)
            
            # Obtener la predicción más común en el historial
            pred_counts = {}
            for p, conf in prediction_history:
                pred_counts[p] = pred_counts.get(p, 0) + conf  # Ponderar por confianza
            
            if pred_counts:
                best_pred = max(pred_counts.items(), key=lambda x: x[1])[0]
                # Solo actualizar si hay suficiente confianza
                if pred_counts[best_pred] >= (MIN_CONSECUTIVE_FRAMES * CONFIDENCE_THRESHOLD):
                    prediction = best_pred
    else:
        # Si no se alcanza el umbral de confianza, no hacer predicción
        prediction = ""
    
    # Limpiar la predicción si no hay manos
    if not results.multi_hand_landmarks:
        prediction = ""
        prediction_history = []  # Limpiar historial cuando no hay manos
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

        
        # Mostrar las 3 mejores predicciones en la pantalla
        y_offset = 50
        for i, (cls, conf) in enumerate(top_predictions, 1):
            text = f"{i}. {cls}: {conf:.1f}%"
            # Resaltar la predicción con mayor confianza
            color = (0, 255, 0) if i == 1 and conf >= CONFIDENCE_THRESHOLD * 100 else (255, 255, 255)
            thickness = 2 if i == 1 and conf >= CONFIDENCE_THRESHOLD * 100 else 1
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness, cv2.LINE_AA)
            y_offset += 25  # Espacio entre líneas
            
        # Mostrar la predicción actual en la parte inferior
        if prediction:
            cv2.putText(frame, f"Prediccion: {prediction}", (10, frame.shape[0] - 20), 
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
