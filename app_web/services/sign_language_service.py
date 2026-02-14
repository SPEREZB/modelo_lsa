# app_web/services/sign_language_service.py
import json
import os
import numpy as np
from typing import List, Dict, Any

class SignLanguageService:
    def __init__(self):
        # Cargar las secuencias de señas
        self.sign_sequences = self._load_sign_sequences()
        
        # Configuración
        self.fps = 30  # Fotogramas por segundo
        self.pause_frames = 10  # Frames de pausa entre palabras
        
    def _load_sign_sequences(self) -> Dict[str, List[List[float]]]:
        """Carga las secuencias de señas desde el archivo JSON"""
        sequences_path = os.path.join(
            os.path.dirname(__file__), 
            '../models/sign_sequences.json'
        )
        with open(sequences_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def text_to_sign_sequence(self, text: str) -> Dict[str, Any]:
        """
        Convierte texto a una secuencia de animación para el avatar
        
        Args:
            text: Texto a convertir en secuencia de señas
            
        Returns:
            Dict con la secuencia de animación y metadatos
        """
        words = self._preprocess_text(text)
        animation_frames = []
        
        for word in words:
            if word in self.sign_sequences:
                # Añadir la secuencia de la palabra
                animation_frames.extend(self.sign_sequences[word])
                # Añadir pausa entre palabras
                animation_frames.extend(self._get_rest_pose())
        
        return {
            'fps': self.fps,
            'keypoints': animation_frames,
            'duration': len(animation_frames) / self.fps
        }
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocesa el texto para buscar coincidencias"""
        # Convertir a minúsculas y dividir en palabras
        words = text.lower().split()
        processed_words = []
        
        # Intentar encontrar frases compuestas primero
        i = 0
        while i < len(words):
            found = False
            # Buscar frases de hasta 3 palabras
            for j in range(min(3, len(words) - i), 0, -1):
                phrase = ' '.join(words[i:i+j])
                if phrase in self.sign_sequences:
                    processed_words.append(phrase)
                    i += j
                    found = True
                    break
            
            if not found:
                # Si no se encontró una frase, buscar palabras individuales
                word = words[i]
                processed_words.append(word if word in self.sign_sequences else '')
                i += 1
        
        return [w for w in processed_words if w]
    
    def _get_rest_pose(self) -> List[List[float]]:
        """Devuelve la postura de reposo"""
        # Usar una secuencia de ceros como postura de reposo
        return [[0.0] * 3 * 21 * 2 for _ in range(self.pause_frames)]
    
    def get_available_signs(self) -> List[str]:
        """Devuelve la lista de señas disponibles"""
        return list(self.sign_sequences.keys())

# Instancia global
sign_service = SignLanguageService()