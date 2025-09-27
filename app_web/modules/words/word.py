# app_web/modules/words/word.py
from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declared_attr

# Importar Base de database para evitar importación circular
from app_web.database import Base

class Word(Base):
    """Modelo para la tabla de palabras del lenguaje de señas."""
    __tablename__ = 'palabras'
    
    id = Column(Integer, primary_key=True, index=True)
    palabra = Column(String(100), unique=True, nullable=False, index=True)
    descripcion = Column(String(500), nullable=True)
    fecha_creacion = Column(DateTime, server_default=func.now())
    fecha_actualizacion = Column(DateTime, onupdate=func.now())
    
    def to_dict(self):
        """Convierte el objeto a un diccionario para serialización JSON."""
        return {
            'id': self.id,
            'palabra': self.palabra,
            'descripcion': self.descripcion,
            'fecha_creacion': self.fecha_creacion.isoformat() if self.fecha_creacion else None,
            'fecha_actualizacion': self.fecha_actualizacion.isoformat() if self.fecha_actualizacion else None
        }
