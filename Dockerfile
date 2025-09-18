# Usar una imagen base de Python
FROM python:3.10.0

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Crear directorio para modelos
RUN mkdir -p /app/models

# Copiar la aplicación y los modelos
COPY . .

# Asegurarse de que los modelos estén en la ubicación correcta
RUN if [ -d "/app/app/models" ]; then \
    mv /app/app/models/* /app/models/ && \
    rmdir /app/app/models; \
    fi

# Puerto expuesto
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:create_app()"]