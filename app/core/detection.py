import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Optional

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

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