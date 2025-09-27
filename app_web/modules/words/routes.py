from flask import Blueprint, request, jsonify
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from ...database import get_db
from .word import Word

words_bp = Blueprint('words', __name__, url_prefix='/api/words')

@words_bp.route('', methods=['POST'])
def create_word():
    """Crear una nueva palabra en el diccionario de señas."""
    data = request.get_json()
    
    if not data or 'palabra' not in data:
        return jsonify({'error': 'Se requiere el campo "palabra"'}), 400
    
    db: Session = next(get_db())
    
    # Verificar si la palabra ya existe
    existing_word = db.query(Word).filter(Word.palabra == data['palabra'].strip().lower()).first()
    if existing_word:
        return jsonify({
            'error': 'La palabra ya existe',
            'palabra': existing_word.to_dict()
        }), 409
    
    try:
        # Crear nueva palabra
        new_word = Word(
            palabra=data['palabra'].strip().lower(),
            descripcion=data.get('descripcion', '').strip()
        )
        
        db.add(new_word)
        db.commit()
        db.refresh(new_word)
        
        return jsonify({
            'message': 'Palabra creada exitosamente',
            'palabra': new_word.to_dict()
        }), 201
        
    except IntegrityError:
        db.rollback()
        return jsonify({'error': 'Error al crear la palabra'}), 500

@words_bp.route('', methods=['GET'])
def get_words():
    """Obtener todas las palabras del diccionario."""
    db: Session = next(get_db())
    
    # Parámetros de paginación
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search = request.args.get('search', '').strip()
    
    # Consulta base
    query = db.query(Word)
    
    # Aplicar búsqueda si se proporciona
    if search:
        query = query.filter(Word.palabra.ilike(f'%{search}%'))
    
    # Paginación
    words = query.order_by(Word.palabra).offset((page - 1) * per_page).limit(per_page).all()
    total = query.count()
    
    return jsonify({
        'data': [word.to_dict() for word in words],
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': (total + per_page - 1) // per_page
        }
    })

@words_bp.route('/<int:word_id>', methods=['GET'])
def get_word(word_id: int):
    """Obtener una palabra por su ID."""
    db: Session = next(get_db())
    
    word = db.query(Word).filter(Word.id == word_id).first()
    
    if not word:
        return jsonify({'error': 'Palabra no encontrada'}), 404
    
    return jsonify(word.to_dict())

@words_bp.route('/<int:word_id>', methods=['PUT'])
def update_word(word_id: int):
    """Actualizar una palabra existente."""
    data = request.get_json()
    db: Session = next(get_db())
    
    word = db.query(Word).filter(Word.id == word_id).first()
    
    if not word:
        return jsonify({'error': 'Palabra no encontrada'}), 404
    
    try:
        # Actualizar campos si se proporcionan
        if 'palabra' in data:
            # Verificar si ya existe otra palabra con el mismo nombre
            existing = db.query(Word).filter(
                Word.palabra == data['palabra'].strip().lower(),
                Word.id != word_id
            ).first()
            
            if existing:
                return jsonify({
                    'error': 'Ya existe otra palabra con ese nombre',
                    'palabra': existing.to_dict()
                }), 409
                
            word.palabra = data['palabra'].strip().lower()
            
        if 'descripcion' in data:
            word.descripcion = data['descripcion'].strip()
        
        db.commit()
        db.refresh(word)
        
        return jsonify({
            'message': 'Palabra actualizada exitosamente',
            'palabra': word.to_dict()
        })
        
    except Exception as e:
        db.rollback()
        return jsonify({'error': 'Error al actualizar la palabra'}), 500

@words_bp.route('/<int:word_id>', methods=['DELETE'])
def delete_word(word_id: int):
    """Eliminar una palabra del diccionario."""
    db: Session = next(get_db())
    
    word = db.query(Word).filter(Word.id == word_id).first()
    
    if not word:
        return jsonify({'error': 'Palabra no encontrada'}), 404
    
    try:
        db.delete(word)
        db.commit()
        
        return jsonify({
            'message': 'Palabra eliminada exitosamente',
            'palabra_id': word_id
        })
        
    except Exception as e:
        db.rollback()
        return jsonify({'error': 'Error al eliminar la palabra'}), 500
