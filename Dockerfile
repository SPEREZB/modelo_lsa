FROM python:3.10.0

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para caché
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código
COPY . .

# Crear directorio para modelos
RUN mkdir -p /app/models

# Variables de entorno
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
 
CMD gunicorn --bind 0.0.0.0:$PORT -w 1 --timeout 3000 run:app