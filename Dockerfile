FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU version from PyTorch's own index
RUN pip install --no-cache-dir \
    torch==2.10.0 \
    torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p artifacts/model_trainer templates

# Fix Ultralytics config directory permission issue
ENV YOLO_CONFIG_DIR=/tmp

# Render sets PORT dynamically, default to 8080 for local use
ENV PORT=8080

EXPOSE 8080

# Use shell form so $PORT variable is expanded at runtime
CMD uvicorn app:app --host 0.0.0.0 --port $PORT