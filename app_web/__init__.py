from flask import Flask
from .routes import main
from .database import init_db

def create_app():
    # Crear la aplicación Flask
    app = Flask(__name__)
    
    # Configuración de la aplicación
    app.config['JSON_SORT_KEYS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
    
    # Inicializar la base de datos
    with app.app_context():
        init_db()
    
    # Registrar blueprints
    app.register_blueprint(main)
    
    return app