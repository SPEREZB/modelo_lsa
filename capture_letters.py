import cv2
import os
import numpy as np
import json
import datetime
from constants import KEYPOINTS_PATH, DATA_PATH, KEYPOINTS_JSON_PATH, WordsConfig
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
WORDS = WordsConfig.WORDS

# FRASES DE EJEMPLO
# EL ALUMNO LE GUSTA LEER
# HOLA como estas?
# EL LE GUSTA COMER 
# EL LE GUSTA LEER
 
 
def create_directories():
    os.makedirs(KEYPOINTS_PATH, exist_ok=True) 
    os.makedirs(KEYPOINTS_JSON_PATH, exist_ok=True) 
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

def convert_to_serializable(obj):
    """Convierte objetos NumPy a tipos nativos de Python"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj

def save_keypoints_json(sequence, label, sample_num):
    """Guarda los keypoints en un archivo JSON con formato específico para el avatar"""
    # Crear nombre de archivo con formato: letra_AAAAMMDD_HHMMSS.json
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(KEYPOINTS_JSON_PATH, f"{label.lower()}_{timestamp}.json")
    
    # Estructura del JSON para el avatar
    data = {
        'metadata': {
            'label': label,
            'frames': len(sequence) if hasattr(sequence, '__len__') else 1,
            'timestamp': timestamp,
            'version': '1.0'
        },
        'keypoints': convert_to_serializable(sequence)
    }
    
    # Asegurar que el directorio existe
    os.makedirs(KEYPOINTS_JSON_PATH, exist_ok=True)
    
    # Guardar el archivo
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Keypoints guardados en: {output_file}")
    return output_file

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
        # Guardar la etiqueta como string UTF-8 para soportar caracteres como ñ y acentos
        str_dtype = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('label', data=label, dtype=str_dtype)
    print(f"Datos guardados en: {output_file} - Forma: {sequence.shape}")
    return output_file

def capture_label(label, sample_num, output_format, capture_limit=None):
    cap = cv2.VideoCapture(0)
    frames = []
    recording = False
    frame_count = 0
    limit = capture_limit or SEQUENCE_LENGTH

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
            frame_count += 1
            print(f"Capturando frame {frame_count}/{limit}", end='\r')

        if recording:
            cv2.putText(frame, f"Grabando: {label} ({len(frames)}/{limit})",
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

        if recording and len(frames) >= limit:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if frames:
        if output_format == 'h5':
            output_file = save_keypoints_h5(frames, label, sample_num)
        else:
            output_file = save_keypoints_json(frames, label, sample_num)
        print(f"\nCaptura completada. Keypoints guardados en: {output_file}")
    else:
        print("No se capturaron datos para guardar.")

def find_missing_samples(files, label):
    """Encuentra los números de muestra faltantes en la secuencia"""
    # Extraer los números de muestra de los archivos existentes
    sample_numbers = []
    prefix = f"{label.lower().replace(' ','_')}_"
    
    for f in files:
        if f.startswith(prefix) and f.endswith('.h5'):
            try:
                num = int(f[len(prefix):-3])  # Extraer número del nombre del archivo
                sample_numbers.append(num)
            except ValueError:
                continue
    
    if not sample_numbers:
        return list(range(1, 51))  # Si no hay muestras, todas faltan
    
    # Encontrar huecos en la secuencia
    max_sample = max(sample_numbers)
    all_samples = set(range(1, max_sample + 1))
    existing_samples = set(sample_numbers)
    missing = sorted(list(all_samples - existing_samples))
    
    return missing

def main():
    create_directories()
    MAX_SAMPLES = 50
    print("Seleccione el formato de salida:")
    print("1. JSON")
    print("2. H5")
    fmt_opt = input("Ingrese 1 o 2: ").strip()
    output_format = 'h5' if fmt_opt == '2' else 'json'
    json_capture_limit = None
    if output_format == 'json':
        print("Seleccione la cantidad de frames para JSON:")
        print("1. 10 frames")
        print("2. 107 frames")
        json_opt = input("Ingrese 1 o 2: ").strip()
        json_capture_limit = 10 if json_opt == '1' else SEQUENCE_LENGTH

    while True:
        label = input("\nIngrese la letra (a-z) o palabra de la lista o 'salir' para terminar: ").lower()
        if label == "salir":
            print("Finalizando grabación.")
            break

        if (len(label) == 1 and label.isalpha()) or (label in WORDS):
            if output_format == 'h5':
                files = [f for f in os.listdir(KEYPOINTS_PATH) if f.startswith(f"{label.lower().replace(' ','_')}_") and f.endswith('.h5')]
                current_samples = len(files)
                missing_samples = find_missing_samples(files, label)
                if current_samples >= MAX_SAMPLES and not missing_samples:
                    print(f"\n✅ Ya hay {MAX_SAMPLES} muestras .h5 para {label} y no faltan muestras.")
                    print("Puede seguir grabando en formato JSON.")
                    use_json = input("¿Desea grabar muestras adicionales en JSON? (s/n): ").lower()
                    if use_json == 's':
                        print("Seleccione la cantidad de frames para JSON:")
                        print("1. 10 frames")
                        print("2. 107 frames")
                        json_opt = input("Ingrese 1 o 2: ").strip()
                        json_capture_limit = 10 if json_opt == '1' else SEQUENCE_LENGTH
                        print(f"\n--- {label.upper()} (JSON) ---")
                        sample_num = 1
                        while True:
                            capture_label(label, sample_num, 'json', capture_limit=json_capture_limit)
                            cont = input(f"¿Desea grabar otra muestra JSON para {label}? (s/n): ")
                            if cont.lower() != 's':
                                break
                    continue
                print(f"\n--- {label.upper()} ---")
                print(f"Muestras existentes: {current_samples}/{MAX_SAMPLES}")
                if missing_samples:
                    print(f"\n⚠️  Muestras faltantes: {', '.join(map(str, missing_samples))}")
                    fill_gap = input(f"¿Desea grabar las muestras faltantes para {label}? (s/n): ").lower()
                    if fill_gap == 's':
                        for sample_num in missing_samples:
                            print(f"\nGrabando muestra faltante #{sample_num}...")
                            capture_label(label, sample_num, output_format)
                            if sample_num < MAX_SAMPLES:
                                cont = input(f"¿Desea grabar la siguiente muestra faltante? (s/n): ")
                                if cont.lower() != 's':
                                    break
                        continue
                if current_samples < MAX_SAMPLES:
                    print(f"\nIniciando grabación de muestras nuevas para {label}...")
                    start_num = max([int(f.split('_')[-1].split('.')[0]) for f in files] + [0]) + 1 if files else 1
                    for sample_num in range(start_num, MAX_SAMPLES + 1):
                        capture_label(label, sample_num, output_format)
                        if sample_num < MAX_SAMPLES:
                            cont = input(f"¿Desea grabar otra muestra para {label}? (s/n): ")
                            if cont.lower() != 's':
                                break
            else:
                print(f"\n--- {label.upper()} ---")
                sample_num = 1
                while True:
                    capture_label(label, sample_num, output_format, capture_limit=json_capture_limit)
                    cont = input(f"¿Desea grabar otra muestra para {label}? (s/n): ")
                    if cont.lower() != 's':
                        break
        else:
            print("❌ Entrada inválida. Solo letras a-z o palabras de la lista.")

if __name__ == "__main__":
    main()
