import numpy as np
from typing import Tuple, List, Optional
from tensorflow.keras.models import Model
from .detection import HandDetector, FaceDetector, extract_keypoints
from app.config import Config

class LSPPredictor:
    def __init__(self, 
                 model: Model,
                 sequence_length: int = 90,
                 min_confidence: float = 0.2,
                 min_consecutive_frames: int = 3,
                 num_features: int = 1530): 
        self.model = model
        self.sequence_length = sequence_length
        self.min_confidence = min_confidence
        self.min_consecutive_frames = min_consecutive_frames
        self.num_features = num_features
        
        # Inicializar detectores
        self.hand_detector = HandDetector()
        self.face_detector = FaceDetector()
        
        # Estado de la predicción
        self.sequence: List[np.ndarray] = []
        self.prediction_history: List[str] = []
        self.last_prediction: str = ""
        
        # Clases del modelo
        self.classes = self._load_classes()
    
    def _load_classes(self) -> List[str]:
        """Carga las clases desde la configuración."""
        config = Config()
        return config.CLASSES
    
    def process_frame(self, 
                     frame: np.ndarray, 
                     include_face: bool = False) -> Tuple[Optional[str], float]:
     
        if frame is None or self.model is None:
            return None, 0.0
        
        # Detectar manos y cara
        hand_results = self.hand_detector.detect(frame)
        face_results = self.face_detector.detect(frame) if include_face else None
        
        # Extraer keypoints
        keypoints = extract_keypoints(
            hand_results, 
            face_results, 
            include_face=include_face,
            num_features=self.num_features
        )
        
        # Actualizar secuencia
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-self.sequence_length:]
        
        # Solo hacer predicción si tenemos suficientes frames
        if len(self.sequence) >= self.sequence_length:
            seq_array = np.array(self.sequence[-self.sequence_length:], dtype=np.float32)
            seq_array = np.expand_dims(seq_array, axis=0)
            
            # Hacer predicción
            pred = self.model.predict(seq_array, verbose=0)[0]
            pred_idx = np.argmax(pred)
            confidence = float(pred[pred_idx])
            
            if 0 <= pred_idx < len(self.classes) and confidence >= self.min_confidence:
                current_pred = self.classes[pred_idx]
                self.prediction_history.append(current_pred)
                
                if len(self.prediction_history) > self.min_consecutive_frames:
                    self.prediction_history.pop(0)
                    
                    # Verificar consenso
                    if all(p == current_pred for p in self.prediction_history):
                        if current_pred != self.last_prediction:
                            self.last_prediction = current_pred
                            return current_pred, confidence
        
        return self.last_prediction, 0.0
    
    def reset(self) -> None:
        """Reinicia el estado de la predicción."""
        self.sequence = []
        self.prediction_history = []
        self.last_prediction = ""
    
    def close(self) -> None:
        """Libera los recursos."""
        self.hand_detector.close()
        self.face_detector.close()