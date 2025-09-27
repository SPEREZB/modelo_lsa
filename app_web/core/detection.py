import os
import cv2
import numpy as np
import mediapipe as mp
import h5py
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

# Constantes
SEQUENCE_LENGTH = 107  # Número fijo de frames por muestra
NUM_HANDS_KEYPOINTS = 126  # 21 landmarks * 3 (x,y,z) * 2 manos
NUM_FACE_KEYPOINTS = 1404  # 468 landmarks * 3 (x,y,z)
DEFAULT_OUTPUT_DIR = "captured_data"
DEFAULT_MODEL_PATH = "models"

# Palabras permitidas
WORDS = [
    "adios", "alumno", "aprender", "bien", "chau", "cocinar", "comer", "comoestas", 
    "dormir", "el", "estudiar", "gracias", "hola", "informe", "investigar", "leer", 
    "legusta", "mellamo", "nolegusta", "perder", "perdon", "tienesrazon", "timido", "yo"
]

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

class HandDetector:
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        
        self.hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect(self, image: np.ndarray) -> any:
       
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.hands.process(image_rgb)
    
    def close(self):
        """Libera los recursos del detector."""
        self.hands.close()


class FaceDetector:
    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
      
        self.face_mesh = mp_face.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect(self, image: np.ndarray) -> any:
                
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(image_rgb)
    
    def close(self):
        self.face_mesh.close()


def extract_keypoints(hand_results: any, 
                     face_results: Optional[any] = None, 
                     include_face: bool = False,
                     num_features: int = 1530) -> np.ndarray:
  
    keypoints = []

    # Extraer keypoints de manos
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
    
    # Extraer keypoints de la cara si está habilitado
    if include_face and face_results and face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
    
    # Asegurar que tenemos el número correcto de características
    if len(keypoints) < num_features:
        keypoints.extend([0.0] * (num_features - len(keypoints)))
    elif len(keypoints) > num_features:
        keypoints = keypoints[:num_features]
    
    return np.array(keypoints, dtype=np.float32)


class LSACapture:
    """Clase para capturar y procesar secuencias de lenguaje de señas."""
    
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR, model_path: str = DEFAULT_MODEL_PATH):
        """Inicializa el capturador de LSA.
        
        Args:
            output_dir: Directorio donde se guardarán las capturas
            model_path: Directorio donde se guardarán los modelos
        """
        self.output_dir = output_dir
        self.model_path = model_path
        self.include_face = False
        self.hand_detector = None
        self.face_detector = None
        self.cap = None
        
        # Crear directorios si no existen
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "frames"), exist_ok=True)
    
    def initialize_detectors(self, include_face: bool = False):
        """Inicializa los detectores de manos y cara.
        
        Args:
            include_face: Si es True, también se detectará el rostro
        """
        self.include_face = include_face
        self.hand_detector = HandDetector(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        if self.include_face:
            self.face_detector = FaceDetector(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
    
    def extract_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """Extrae los puntos clave de las manos y opcionalmente de la cara.
        
        Args:
            frame: Imagen de entrada en formato BGR
            
        Returns:
            frame: Imagen con los landmarks dibujados
            keypoints: Array de numpy con los puntos clave normalizados
        """
        if self.hand_detector is None:
            self.initialize_detectors(self.include_face)
        
        # Detectar manos y cara
        hand_results = self.hand_detector.detect(frame)
        face_results = self.face_detector.detect(frame) if self.include_face else None
        
        keypoints = []
        
        # Extraer keypoints de manos
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
        
        # Asegurar que tenemos el número correcto de características de manos
        if len(keypoints) < NUM_HANDS_KEYPOINTS:
            keypoints.extend([0.0] * (NUM_HANDS_KEYPOINTS - len(keypoints)))
        
        # Extraer keypoints de la cara si está habilitado
        if self.include_face and face_results and face_results.multi_face_landmarks:
            face_points = []
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_TESSELATION)
                for lm in face_landmarks.landmark:
                    face_points.extend([lm.x, lm.y, lm.z])
            
            # Asegurar que tenemos el número correcto de características de cara
            if len(face_points) < NUM_FACE_KEYPOINTS:
                face_points.extend([0.0] * (NUM_FACE_KEYPOINTS - len(face_points)))
            
            keypoints.extend(face_points)
        
        return frame, np.array(keypoints, dtype=np.float32)
    
    def capture_sequence(self, label: str, sample_num: int = 1):
        """Captura una secuencia de frames para una etiqueta dada.
        
        Args:
            label: Etiqueta de la secuencia (letra o palabra)
            sample_num: Número de muestra para la etiqueta
        """
        if self.hand_detector is None:
            self.initialize_detectors(self.include_face)
        
        self.cap = cv2.VideoCapture(0)
        frames = []
        recording = False
        
        print(f"Presiona 's' para empezar a grabar: {label}")
        print("Presiona 'q' para terminar la grabación")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame, keypoints = self.extract_keypoints(frame)
            
            if recording:
                frames.append(keypoints)
            
            # Mostrar información en el frame
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
        
        # Guardar los datos capturados
        if frames:
            self._save_sequence(frames, label, sample_num)
        else:
            print("No se capturaron datos para guardar.")
        
        # Liberar recursos
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _save_sequence(self, sequence: List[np.ndarray], label: str, sample_num: int):
        """Guarda la secuencia de keypoints en un archivo HDF5.
        
        Args:
            sequence: Lista de arrays de keypoints
            label: Etiqueta de la secuencia
            sample_num: Número de muestra
        """
        # Asegurar que la secuencia tenga la longitud correcta
        if len(sequence) < SEQUENCE_LENGTH:
            padding = np.zeros((SEQUENCE_LENGTH - len(sequence), len(sequence[0])), dtype=np.float32)
            sequence = np.vstack([sequence, padding])
        else:
            sequence = np.array(sequence[:SEQUENCE_LENGTH], dtype=np.float32)
        
        # Crear nombre de archivo único
        output_file = os.path.join(self.output_dir, f"{label.lower().replace(' ', '_')}_{sample_num}.h5")
        
        # Guardar en formato HDF5
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('keypoints', data=sequence)
            f.create_dataset('label', data=np.string_(label))
        
        print(f"Datos guardados en: {output_file} - Forma: {sequence.shape}")
    
    def close(self):
        """Libera los recursos de los detectores y la cámara."""
        if self.hand_detector:
            self.hand_detector.close()
        if self.face_detector:
            self.face_detector.close()
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Función principal para ejecutar la captura de secuencias LSA."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Capturador de secuencias LSA')
    parser.add_argument('--face', action='store_true', help='Incluir detección de rostro')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR, 
                       help=f'Directorio de salida (por defecto: {DEFAULT_OUTPUT_DIR})')
    args = parser.parse_args()
    
    capture = LSACapture(output_dir=args.output)
    
    try:
        capture.initialize_detectors(include_face=args.face)
        
        while True:
            label = input("\nIngrese la letra (a-z) o palabra de la lista o 'salir' para terminar: ").lower()
            
            if label == "salir":
                print("Finalizando grabación.")
                break
            
            if (len(label) == 1 and label.isalpha()) or (label in WORDS):
                # Contar muestras existentes
                files = [f for f in os.listdir(capture.output_dir) 
                        if f.startswith(f"{label.lower().replace(' ', '_')}_") and f.endswith('.h5')]
                current_samples = len(files)
                
                # Capturar nueva muestra
                capture.capture_sequence(label, current_samples + 1)
                
                # Preguntar si desea grabar otra muestra
                cont = input(f"¿Desea grabar otra muestra para {label}? (s/n): ")
                if cont.lower() != 's':
                    continue
            else:
                print("Entrada inválida. Solo letras a-z o palabras de la lista.")
    
    except KeyboardInterrupt:
        print("\nCaptura interrumpida por el usuario.")
    finally:
        capture.close()


if __name__ == "__main__":
    main()