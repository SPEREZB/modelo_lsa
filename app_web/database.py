# app_web/database.py
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de la base de datos
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:123@localhost:5432/postgres")

# Crear motor de SQLAlchemy
engine = create_engine(DATABASE_URL)
metadata = MetaData()
Base = declarative_base(metadata=metadata)

# Configurar sesión
SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)

# Dependencia para inyectar la sesión
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Importar y registrar modelos después de definir Base
def register_models():
    # Importar modelos aquí evita importaciones circulares
    from app_web.modules.words.word import Word
    return [Word]  # Agregar aquí otros modelos en el futuro

# Crear tablas
def init_db():
    # Registrar todos los modelos primero
    register_models()
    # Luego crear las tablas
    Base.metadata.create_all(bind=engine)