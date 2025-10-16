# --- Stage 1: Build Stage ---
# Use an official NVIDIA CUDA runtime image as a parent image.
# This image contains the CUDA toolkit and cuDNN, which are required for GPU acceleration.
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV PYTHONUNBUFFERED=1  

# Install system dependencies
# - python3.10 and pip: For running the application.
# - ffmpeg: Required by many audio processing libraries, including faster-whisper.
# - git: For version control and potentially fetching dependencies.
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install PyTorch with CUDA support first. This is crucial for GPU acceleration.
# The index-url points to the official PyTorch wheel repository for CUDA 12.1.
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the Python dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY main_vad.py .

# Create a directory for persistent data
RUN mkdir -p /app/data

# Expose the port the app runs on
EXPOSE 8000

# Set the command to run when the container starts.
# Use uvicorn to run the FastAPI application.
# --host 0.0.0.0 makes the server accessible from outside the container.
CMD ["uvicorn", "main_vad:app", "--host", "0.0.0.0", "--port", "8000"]
