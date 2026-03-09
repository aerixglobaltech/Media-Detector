# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install system dependencies required for OpenCV, PyTorch, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install PyTorch for CPU explicitly to save space and avoid CUDA drivers on Railway
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set cache directories for large models out of the root directory
ENV DEEPFACE_HOME=/app/.cache
ENV TORCH_HOME=/app/.cache
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN mkdir -p /app/.cache && chmod 777 /app/.cache && \
    mkdir -p /tmp/Ultralytics && chmod 777 /tmp/Ultralytics

# Copy the rest of the application code
COPY . .

# Install Waitress for production serving mode (instead of Flask Dev Server)
RUN pip install --no-cache-dir waitress

# Expose port 5000 for the Flask app
EXPOSE 5000

ENV PORT=5000

# Start the application using Waitress natively through the python script
CMD ["python", "app.py"]
