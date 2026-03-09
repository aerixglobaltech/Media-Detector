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
ENV YOLOV8_HOME=/app/.cache

# Copy the rest of the application code
COPY . .

# Expose port 5000 for the Flask app
EXPOSE 5000

# Railway will provide a PORT environment variable, Flask needs to listen to it or we just force it to 5000
# and map Railway to 5000. In app.py it binds to 0.0.0.0:5000, which is perfect for Railway if we set PORT=5000.
ENV PORT=5000

# Start the Flask interface
CMD ["python", "app.py"]
