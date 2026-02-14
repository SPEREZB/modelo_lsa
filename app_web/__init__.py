from flask import Flask
from flask_cors import CORS
import os
from .routes import main, global_predictor, socketio
from .database import init_db
from app_web.controllers.avatar_controller import avatar_bp

def create_app():
    # Crear la aplicación Flask
    app = Flask(__name__)
    CORS(app)
    
    # Configuración de la aplicación
    app.config['JSON_SORT_KEYS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
    
    # Configurar el predictor en la aplicación
    app.config['predictor'] = global_predictor
    
    # Inicializar SocketIO
    socketio.init_app(app, cors_allowed_origins="*")
    
    # Inicializar la base de datos
    with app.app_context():
        init_db()
    
    # Registrar blueprints
    app.register_blueprint(main)
    app.register_blueprint(avatar_bp, url_prefix='/api/avatar')
    return app, socketio