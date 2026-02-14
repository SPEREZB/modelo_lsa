import numpy as np
import time
from typing import Tuple, List, Optional
from tensorflow.keras.models import Model
from .detection import HandDetector, FaceDetector, extract_keypoints
from app_web.config import Config


class LSPPredictor:
    def __init__(
        self,
        model: Model,
        sequence_length: int = 107,
        num_features: int = 126,  # 21 puntos * 3 coords * 2 manos
        confidence_threshold: float = 0.40,
        motion_threshold: float = 0.004,
        min_gesture_frames: int = 40,
        quiet_frames_to_end: int = 8,
        cooldown_time: float = 1.2,
        require_hands: bool = True,  # Si True, reinicia sin manos
        skip_face: bool = True,  # Optimización: no procesar cara por defecto
    ):
        self.model = model
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.confidence_threshold = confidence_threshold
        self.motion_threshold = motion_threshold
        self.min_gesture_frames = min_gesture_frames
        self.quiet_frames_to_end = quiet_frames_to_end
        self.cooldown_time = cooldown_time
        self.require_hands = require_hands
        self.skip_face = skip_face

        # Detectores - solo inicializar lo necesario
        self.hand_detector = HandDetector(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        # Solo inicializar face_detector si se necesita
        self.face_detector = None if skip_face else FaceDetector()

        self.state: str = "WAITING"
        self.sequence: List[np.ndarray] = []
        self.prev_keypoints: Optional[np.ndarray] = None
        self.cooldown_start: float = 0.0
        self.last_output: Optional[Tuple[str, float]] = None
        self.last_motion: float = 0.0
        self.last_confidence: float = 0.0
        self.quiet_frames: int = 0

        # Clases
        self.classes = self._load_classes()

    def _load_classes(self) -> List[str]:
        config = Config()
        return config.CLASSES

    def _motion_level(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    def _pad_sequence(self, seq: List[np.ndarray]) -> np.ndarray:
        """Asegura que la secuencia tenga `sequence_length` frames, con padding de ceros."""
        if len(seq) == 0:
            return np.zeros((self.sequence_length, self.num_features), dtype=np.float32)
        
        if len(seq) >= self.sequence_length:
            return np.array(seq[-self.sequence_length:], dtype=np.float32)
        
        # Padding al inicio con ceros
        padding = np.zeros((self.sequence_length - len(seq), self.num_features), dtype=np.float32)
        current = np.array(seq, dtype=np.float32)
        return np.vstack([padding, current])

    def process_frame(
        self,
        frame: np.ndarray,
        include_face: bool = False
    ) -> Tuple[Optional[str], float]:
        """
        Procesa un frame individual (desde app móvil).
        Devuelve (predicción, confianza) solo si es estable y confiable.
        """
        if frame is None or self.model is None:
            return None, 0.0

        # Detectar manos (optimizado: no procesar cara a menos que sea necesario)
        hand_results = self.hand_detector.detect(frame)
        face_results = None
        if include_face and self.face_detector is not None:
            face_results = self.face_detector.detect(frame)

        # Si no hay manos y se requieren, reiniciar estado parcialmente
        if self.require_hands and (not hand_results or not hand_results.multi_hand_landmarks):
            self.sequence = []
            self.prev_keypoints = None
            self.state = "WAITING"
            self.quiet_frames = 0
            return None, 0.0

        # Extraer keypoints
        try:
            keypoints = extract_keypoints(
                hand_results=hand_results,
                face_results=face_results,
                include_face=include_face,
                num_features=self.num_features
            )
        except Exception:
            return None, 0.0

        motion = 0.0
        if self.prev_keypoints is not None:
            motion = self._motion_level(self.prev_keypoints, keypoints)
        self.prev_keypoints = keypoints
        self.last_motion = motion

        now = time.time()

        if self.state == "WAITING":
            if motion > self.motion_threshold:
                self.sequence = []
                self.quiet_frames = 0
                self.state = "RECORDING"

        if self.state == "RECORDING":
            self.sequence.append(keypoints)

            if motion <= self.motion_threshold:
                self.quiet_frames += 1
            else:
                self.quiet_frames = 0

            should_predict = (
                len(self.sequence) >= self.sequence_length or
                (len(self.sequence) >= self.min_gesture_frames and self.quiet_frames >= self.quiet_frames_to_end)
            )
            if should_predict:
                self.state = "PREDICTING"

        if self.state == "PREDICTING":
            padded_seq = self._pad_sequence(self.sequence)
            seq_array = np.expand_dims(padded_seq, axis=0)

            try:
                pred_probs = self.model.predict(seq_array, verbose=0)[0]
                pred_idx = int(np.argmax(pred_probs))
                confidence = float(pred_probs[pred_idx])
                self.last_confidence = confidence
                if 0 <= pred_idx < len(self.classes):
                    pred_label = self.classes[pred_idx]
                else:
                    pred_label = None

                if pred_label and confidence >= self.confidence_threshold:
                    self.last_output = (pred_label, confidence)
                else:
                    self.last_output = None
            except Exception:
                self.last_output = None

            self.cooldown_start = now
            self.state = "COOLDOWN"
            self.sequence = []
            self.quiet_frames = 0

            if self.last_output:
                return self.last_output
            return None, 0.0

        if self.state == "COOLDOWN":
            if now - self.cooldown_start > self.cooldown_time:
                self.state = "WAITING"

        return None, 0.0

    def get_debug_status(self) -> dict:
        cooldown_remaining = 0.0
        if self.state == "COOLDOWN":
            cooldown_remaining = max(0.0, self.cooldown_time - (time.time() - self.cooldown_start))

        last_label = self.last_output[0] if self.last_output else None
        last_conf = float(self.last_output[1]) if self.last_output else 0.0

        return {
            "state": self.state,
            "frames": len(self.sequence),
            "sequence_length": int(self.sequence_length),
            "motion": float(self.last_motion),
            "quiet_frames": int(self.quiet_frames),
            "min_gesture_frames": int(self.min_gesture_frames),
            "cooldown_remaining": float(cooldown_remaining),
            "last_label": last_label,
            "last_label_confidence": last_conf,
            "last_confidence": float(self.last_confidence),
        }

    def reset(self) -> None:
        """Reinicia completamente el estado."""
        self.sequence = []
        self.prev_keypoints = None
        self.state = "WAITING"
        self.last_output = None
        self.cooldown_start = 0.0
        self.last_motion = 0.0
        self.last_confidence = 0.0
        self.quiet_frames = 0

    def close(self) -> None:
        """Libera recursos."""
        if self.hand_detector:
            self.hand_detector.close()
        if self.face_detector:
            self.face_detector.close()