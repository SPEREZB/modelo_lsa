# app_web/controllers/avatar_controller.py
from flask import Blueprint, request, jsonify
import base64
import numpy as np
from app_web.services.speech_service import speech_service
import os
from werkzeug.utils import secure_filename
from app_web.services.sign_language_service import sign_service
import json

# Configuración de archivos
UPLOAD_FOLDER = 'app_web/static/uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'webm'}
 
avatar_bp = Blueprint('avatar', __name__)

@avatar_bp.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # Verificar si se envió un archivo de audio
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No se proporcionó ningún archivo de audio'}), 400
        
        audio_file = request.files['audio']
        
        # Validar el archivo
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'No se seleccionó ningún archivo'}), 400
    
        
        # Asegurar carpeta de subida
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Guardar archivo temporal
        filename = secure_filename(audio_file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        audio_file.save(temp_path)
        
        try:
            # Procesar el audio pasando la ruta del archivo
            result = speech_service.process_audio(temp_path)
                
            if not result['success']:
                # Si hay error de dependencia (ffmpeg), devolver 500
                err_msg = (result.get('error') or '').lower()
                status_code = 500 if ('ffmpeg' in err_msg or 'ffprobe' in err_msg or 'decoder' in err_msg) else 400
                return jsonify({
                    'success': False, 
                    'error': result.get('error', 'Error al procesar el audio')
                }), status_code
                
            return jsonify({
                'success': True,
                'text': result.get('text', ''),
                'words': result.get('words', [])
            })
            
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'Error al procesar el audio: {str(e)}'
            }), 500
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Error inesperado: {str(e)}'
        }), 500

@avatar_bp.route('/text_to_sign', methods=['POST'])
def text_to_sign():
    try:
        # Verificar si los datos son JSON
        if not request.is_json:
            return jsonify({
                'success': False, 
                'error': 'El contenido debe ser de tipo application/json'
            }), 400
            
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False, 
                'error': 'No se proporcionó texto'
            }), 400
            
        # Convertir texto a secuencia de animación de señas
        animation_data = sign_service.text_to_sign_sequence(text)
        
        if not animation_data['keypoints']:
            return jsonify({
                'success': False,
                'error': 'No se encontraron señas para el texto proporcionado',
                'available_signs': sign_service.get_available_signs()
            }), 404
        
        return jsonify({
            'success': True,
            'text': text,
            'animation': animation_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Error al procesar el texto: {str(e)}',
            'available_signs': sign_service.get_available_signs()
        }), 500

@avatar_bp.route('/available_signs', methods=['GET'])
def get_available_signs():
    """Devuelve la lista de señas disponibles"""
    try:
        return jsonify({
            'success': True,
            'signs': sign_service.get_available_signs()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error al obtener las señas disponibles: {str(e)}'
        }), 500

@avatar_bp.route('/keypoints', methods=['GET'])
def get_keypoints_first_frame():
    """Devuelve el primer frame de un archivo JSON de keypoints según 'name'.

    Parámetros:
      - name: nombre base del archivo (sin .json). Si no existe, intenta con la primera letra.
    """
    try:
        name = (request.args.get('name') or '').strip().lower()
        if not name:
            return jsonify({'success': False, 'error': 'Parámetro name es requerido'}), 400

        # Obtener la ruta absoluta del directorio del proyecto
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        base_dir = os.path.join(project_root, 'data', 'keypoints_json')
        candidates = []
        # Candidato 1: nombre completo
        candidates.append(os.path.join(base_dir, f"{name}.json"))
        # Candidato 2: primera letra
        if len(name) > 0:
            candidates.append(os.path.join(base_dir, f"{name[0]}.json"))

        target_path = None
        for p in candidates:
            if os.path.exists(p):
                target_path = p
                break

        if not target_path:
            # Listar disponibles para ayudar a depurar
            try:
                available = [f for f in os.listdir(base_dir) if f.lower().endswith('.json')]
            except Exception:
                available = []
            return jsonify({
                'success': False,
                'error': f'No se encontró archivo JSON para "{name}"',
                'available': available
            }), 404

        with open(target_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        keypoints = data.get('keypoints') or []
        if not keypoints or not isinstance(keypoints, list):
            return jsonify({'success': False, 'error': 'Formato de JSON inválido: keypoints no encontrado'}), 500

        first_frame = keypoints[0]
        if isinstance(first_frame, dict) and 'keypoints' in first_frame:
            first_frame = first_frame['keypoints']

        if not isinstance(first_frame, list):
            return jsonify({'success': False, 'error': 'Formato de frame inválido'}), 500

        return jsonify({
            'success': True,
            'name': name,
            'frame': first_frame
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error al leer keypoints: {str(e)}'}), 500

@avatar_bp.route('/keypoints_sequence', methods=['POST'])
def get_keypoints_sequence():
    """Devuelve una secuencia de keypoints para múltiples palabras.
    
    Implementa búsqueda en cascada:
    1. Busca cada palabra por separado
    2. Si alguna no se encuentra, intenta buscar la combinación con guión bajo
    
    Body JSON:
      - words: array de palabras a buscar
    
    Returns:
      - sequence: array de objetos con keypoints para cada palabra/combinación encontrada
    """
    try:
        if not request.is_json:
            return jsonify({
                'success': False, 
                'error': 'El contenido debe ser de tipo application/json'
            }), 400
        
        data = request.get_json()
        words = data.get('words', [])
        
        if not words or not isinstance(words, list):
            return jsonify({
                'success': False,
                'error': 'Se requiere un array de palabras'
            }), 400
        
        # Obtener la ruta del directorio de keypoints
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        base_dir = os.path.join(project_root, 'data', 'keypoints_json')
        
        sequence = []
        not_found = []
        
        # Procesar palabras con búsqueda en cascada
        i = 0
        while i < len(words):
            word = words[i].strip().lower()
            if not word:
                i += 1
                continue
            
            # Intentar buscar la palabra individual
            word_path = os.path.join(base_dir, f"{word}.json")
            
            if os.path.exists(word_path):
                # Palabra encontrada individualmente
                try:
                    with open(word_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    keypoints = data.get('keypoints', [])
                    if keypoints:
                        sequence.append({
                            'word': word,
                            'keypoints': keypoints,
                            'found_as': 'individual'
                        })
                except Exception as e:
                    not_found.append({'word': word, 'error': str(e)})
                i += 1
            else:
                # Palabra no encontrada, intentar combinación con la siguiente
                if i + 1 < len(words):
                    next_word = words[i + 1].strip().lower()
                    combined = f"{word}_{next_word}"
                    combined_path = os.path.join(base_dir, f"{combined}.json")
                    
                    if os.path.exists(combined_path):
                        # Combinación encontrada
                        try:
                            with open(combined_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            keypoints = data.get('keypoints', [])
                            if keypoints:
                                sequence.append({
                                    'word': combined,
                                    'keypoints': keypoints,
                                    'found_as': 'combined',
                                    'original_words': [word, next_word]
                                })
                            i += 2  # Saltar ambas palabras
                        except Exception as e:
                            not_found.append({'word': combined, 'error': str(e)})
                            i += 1
                    else:
                        # Ni individual ni combinada encontrada
                        # Intentar con primera letra como fallback
                        letter_path = os.path.join(base_dir, f"{word[0]}.json")
                        if os.path.exists(letter_path):
                            try:
                                with open(letter_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                keypoints = data.get('keypoints', [])
                                if keypoints:
                                    sequence.append({
                                        'word': word,
                                        'keypoints': keypoints,
                                        'found_as': 'letter',
                                        'letter': word[0]
                                    })
                            except Exception as e:
                                not_found.append({'word': word, 'error': str(e)})
                        else:
                            not_found.append({'word': word, 'error': 'No se encontró archivo JSON'})
                        i += 1
                else:
                    # Es la última palabra y no se encontró, intentar con primera letra
                    letter_path = os.path.join(base_dir, f"{word[0]}.json")
                    if os.path.exists(letter_path):
                        try:
                            with open(letter_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            keypoints = data.get('keypoints', [])
                            if keypoints:
                                sequence.append({
                                    'word': word,
                                    'keypoints': keypoints,
                                    'found_as': 'letter',
                                    'letter': word[0]
                                })
                        except Exception as e:
                            not_found.append({'word': word, 'error': str(e)})
                    else:
                        not_found.append({'word': word, 'error': 'No se encontró archivo JSON'})
                    i += 1
        
        # Listar archivos disponibles
        try:
            available = [f.replace('.json', '') for f in os.listdir(base_dir) if f.lower().endswith('.json')]
        except Exception:
            available = []
        
        return jsonify({
            'success': True,
            'sequence': sequence,
            'not_found': not_found,
            'available': available,
            'total_found': len(sequence),
            'total_not_found': len(not_found)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error al procesar la secuencia: {str(e)}'
        }), 500