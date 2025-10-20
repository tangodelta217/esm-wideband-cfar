# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema mínimas (opcional pero útil)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Instala requirements (sin gnuradio)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copiamos el proyecto
COPY . /app

# Comando por defecto: ejecutar la evaluación
CMD ["python", "-m", "eval.evaluate_cfar"]
