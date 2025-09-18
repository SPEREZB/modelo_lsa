from flask import Blueprint

# Crear el blueprint de la API
api_bp = Blueprint('api', __name__)

# Importar rutas para registrar los endpoints
from . import routes  
