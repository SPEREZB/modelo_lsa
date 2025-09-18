import mediapipe as mp
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class MediaPipeService:
    def __init__(self):
        # Inicializar soluciones de MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils 
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Detecta puntos clave en la imagen usando MediaPipe Holistic."""
        # Convertir BGR a RGB
        image_rgb = image[..., ::-1]
        
        # Hacer la detección
        results = self.holistic.process(image_rgb)
        
        return {
            'pose': results.pose_landmarks,
            'left_hand': results.left_hand_landmarks,
            'right_hand': results.right_hand_landmarks,
            'face': results.face_landmarks
        }

    def extract_keypoints(self, results: Dict[str, Any], include_face: bool = False) -> np.ndarray:
        """Extrae y normaliza los keypoints de los resultados de detección."""
        keypoints = []
        
        # Extraer keypoints de pose (33 landmarks)
        if results['pose']:
            pose = np.array([[res.x, res.y, res.z, res.visibility] 
                           for res in results['pose'].landmark]).flatten()
        else:
            pose = np.zeros(33*4)
        
        # Extraer keypoints de la mano izquierda (21 landmarks)
        if results['left_hand']:
            lh = np.array([[res.x, res.y, res.z] 
                          for res in results['left_hand'].landmark]).flatten()
        else:
            lh = np.zeros(21*3)
            
        # Extraer keypoints de la mano derecha (21 landmarks)
        if results['right_hand']:
            rh = np.array([[res.x, res.y, res.z] 
                          for res in results['right_hand'].landmark]).flatten()
        else:
            rh = np.zeros(21*3)
            
        # Extraer keypoints de la cara (468 landmarks)
        if include_face and results['face']:
            face = np.array([[res.x, res.y, res.z] 
                           for res in results['face'].landmark]).flatten()
        else:
            face = np.zeros(468*3)
            
        # Concatenar todos los keypoints
        return np.concatenate([pose, lh, rh, face])

    def draw_landmarks(self, image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Dibuja los landmarks en la imagen."""
        # Copiar la imagen para no modificar la original
        image_copy = image.copy()
        
        # Dibujar landmarks
        if results['pose']:
            self.mp_drawing.draw_landmarks(
                image_copy, results['pose'], self.mp_holistic.POSE_CONNECTIONS)
            
        if results['left_hand']:
            self.mp_drawing.draw_landmarks(
                image_copy, results['left_hand'], self.mp_holistic.HAND_CONNECTIONS)
            
        if results['right_hand']:
            self.mp_drawing.draw_landmarks(
                image_copy, results['right_hand'], self.mp_holistic.HAND_CONNECTIONS)
            
        if results['face']:
            self.mp_drawing.draw_landmarks(
                image_copy, results['face'], self.mp_holistic.FACEMESH_TESSELATION)
            
        return image_copy

    def close(self):
        self.holistic.close()