# app_web/services/speech_service.py
import speech_recognition as sr
import tempfile
import os
from pydub import AudioSegment
import io
import numpy as np
import re

class SpeechToSignService:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.8
        self.recognizer.non_speaking_duration = 0.3

    def process_audio(self, audio_file_path):
        """
        Convierte audio a texto y retorna array de palabras
        
        Args:
            audio_file_path: Ruta al archivo de audio guardado temporalmente
            
        Returns:
            Dict con success, text (string completo), words (array de palabras)
        """
        try:
            # Leer el archivo de audio
            audio = AudioSegment.from_file(audio_file_path)
            
            # Convertir a formato compatible con speech_recognition
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Exportar a formato WAV en memoria
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            # Crear AudioData para speech_recognition
            with sr.AudioFile(wav_buffer) as source:
                audio_data = self.recognizer.record(source)
            
            # Reconocer el audio usando Google Speech Recognition
            text = self.recognizer.recognize_google(audio_data, language="es-ES")
            
            # Convertir texto a array de palabras
            # Limpiar y dividir el texto en palabras
            words = self._text_to_words_array(text)
            
            return {
                'success': True,
                'text': text,
                'words': words
            }
            
        except sr.UnknownValueError:
            return {
                'success': False,
                'error': 'No se pudo entender el audio'
            }
        except sr.RequestError as e:
            return {
                'success': False,
                'error': f'Error en el servicio de reconocimiento: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error al procesar el audio: {str(e)}'
            }
    
    def _text_to_words_array(self, text):
        """
        Convierte un texto en un array de palabras
        Ejemplo: "hola, como estas?" -> ["hola", "como", "estas"]
        
        Args:
            text: Texto a convertir
            
        Returns:
            Lista de palabras limpias
        """
        if not text:
            return []
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar signos de puntuación pero mantener espacios
        text = re.sub(r'[^\w\s]', '', text)
        
        # Dividir por espacios y filtrar palabras vacías
        words = [word.strip() for word in text.split() if word.strip()]
        
        return words

# Instancia global
speech_service = SpeechToSignService()