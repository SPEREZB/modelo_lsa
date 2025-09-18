import cv2
import os
import numpy as np
from constants import KEYPOINTS_PATH, DATA_PATH
import mediapipe as mp
import h5py

# --- Configuración ---
SEQUENCE_LENGTH = 107  # Número fijo de frames por muestra
NUM_HANDS_KEYPOINTS = 126
NUM_FACE_KEYPOINTS = 1404

# --- Inicializar MediaPipe ---
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

print("Seleccione el modo de detección:")
print("1. Solo manos")
print("2. Manos y rostro")
opcion = input("Ingrese 1 o 2: ")

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

if opcion == "2":
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
else:
    face_mesh = None

TOTAL_KEYPOINTS = NUM_HANDS_KEYPOINTS + (NUM_FACE_KEYPOINTS if face_mesh else 0)

# --- Palabras permitidas ---
WORDS = [
    "adios", "alumno", "bien", "chau", "comer", "comoestas", 
    "dormir", "el", "gracias", "hola", "informe", "investigar", "leer", 
    "legusta", "mellamo", "nolegusta", "perdon", "tiene", "tienesrazon", "timido", "yo"
]

# FRASES DE EJEMPLO
# EL ALUMNO LE GUSTA LEER
# HOLA como estas?
# EL LE GUSTA COMER 
# EL LE GUSTA LEER
 
 
def create_directories():
    os.makedirs(KEYPOINTS_PATH, exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "videos"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "frames"), exist_ok=True)

def normalize_landmarks(landmarks):
    return np.array(landmarks, dtype=np.float32)

def get_keypoints(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(image_rgb)
    results_face = face_mesh.process(image_rgb) if face_mesh else None

    keypoints = []

    # MANOS
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks[:2]:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for point in hand_landmarks.landmark:
                keypoints.extend([point.x, point.y, point.z])

    if len(keypoints) < NUM_HANDS_KEYPOINTS:
        keypoints.extend([0.0] * (NUM_HANDS_KEYPOINTS - len(keypoints)))

    # CARA
    face_points = []
    if results_face and results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_TESSELATION)
            for point in face_landmarks.landmark:
                face_points.extend([point.x, point.y, point.z])

    if face_mesh:
        if len(face_points) < NUM_FACE_KEYPOINTS:
            face_points.extend([0.0] * (NUM_FACE_KEYPOINTS - len(face_points)))
        keypoints.extend(face_points)

    keypoints = normalize_landmarks(keypoints)
    return frame, keypoints

def save_keypoints_h5(sequence, label, sample_num):
    if len(sequence) < SEQUENCE_LENGTH:
        padding = np.zeros((SEQUENCE_LENGTH - len(sequence), TOTAL_KEYPOINTS), dtype=np.float32)
        sequence = np.vstack([sequence, padding])
    elif len(sequence) > SEQUENCE_LENGTH:
        sequence = np.array(sequence[:SEQUENCE_LENGTH], dtype=np.float32)
    else:
        sequence = np.array(sequence, dtype=np.float32)

    output_file = os.path.join(KEYPOINTS_PATH, f"{label.lower().replace(' ','_')}_{sample_num}.h5")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('keypoints', data=sequence)
        f.create_dataset('label', data=np.string_(label))
    print(f"Datos guardados en: {output_file} - Forma: {sequence.shape}")

def capture_label(label, sample_num):
    cap = cv2.VideoCapture(0)
    frames = []
    recording = False

    print(f"Presiona 's' para empezar a grabar: {label}")
    print("Presiona 'q' para terminar la grabación")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame, keypoints = get_keypoints(frame)

        if recording:
            frames.append(keypoints)

        if recording:
            cv2.putText(frame, f"Grabando: {label} ({len(frames)}/{SEQUENCE_LENGTH})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Presiona 's' para grabar: {label}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Captura de Secuencias LSA', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            recording = True
        elif key == ord('q'):
            break

        if recording and len(frames) >= SEQUENCE_LENGTH:
            print(f"Grabación completa para {label}")
            break

    cap.release()
    cv2.destroyAllWindows()

    if frames:
        save_keypoints_h5(frames, label, sample_num)
    else:
        print("No se capturaron datos para guardar.")

def main():
    create_directories()
    MAX_SAMPLES = 50

    while True:
        label = input("\nIngrese la letra (a-z) o palabra de la lista o 'salir' para terminar: ").lower()
        if label == "salir":
            print("Finalizando grabación.")
            break

        if (len(label) == 1 and label.isalpha()) or (label in WORDS):
            files = [f for f in os.listdir(KEYPOINTS_PATH) if f.startswith(f"{label.lower().replace(' ','_')}_") and f.endswith('.h5')]
            current_samples = len(files)

            if current_samples >= MAX_SAMPLES:
                print(f"Ya hay {MAX_SAMPLES} muestras para {label}.")
                continue

            print(f"\n--- Grabando: {label} (Actualmente {current_samples} muestras) ---")
            for sample_num in range(current_samples + 1, MAX_SAMPLES + 1):
                capture_label(label, sample_num)
                if sample_num < MAX_SAMPLES:
                    cont = input(f"¿Desea grabar otra muestra para {label}? (s/n): ")
                    if cont.lower() != 's':
                        break
        else:
            print("Entrada inválida. Solo letras a-z o palabras de la lista.")

if __name__ == "__main__":
    main()
