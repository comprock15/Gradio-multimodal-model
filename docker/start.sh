#!/bin/bash

set -a
source docker/.env
set +a

# Runtime configuration (not in .env)
DEVICE=${1:-cuda} # cuda/cpu
MODEL_SIZE=${2:-256M} # 256M/500M
PORT=${3:-7865}

echo "Runtime configuration:"
echo "DEVICE: $DEVICE"
echo "MODEL_SIZE: $MODEL_SIZE"
echo "PORT: $PORT"

# Create volumes if they don't exist
echo "Creating volumes for models and cache..."
docker volume create $MODEL_VOLUME 2>/dev/null || true
docker volume create $CACHE_VOLUME 2>/dev/null || true

# Check if NVIDIA GPU is available
if [ "$DEVICE" = "cuda" ] && command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Running with GPU support..."
    GPU_FLAGS="--gpus all"
    DEVICE="cuda"
else
    echo "Running on CPU..."
    GPU_FLAGS=""
    DEVICE="cpu"
fi

# Run the container
echo "Starting SmolVLM2 application..."
docker run -d \
    --name $CONTAINER_NAME \
    --ipc=host \
    $GPU_FLAGS \
    -p $PORT:7860 \
    -v $MODEL_VOLUME:/home/$USER/smolvlm2/models \
    -v $CACHE_VOLUME:/home/$USER/smolvlm2/cache \
    -e DEVICE=$DEVICE \
    -e MODEL_SIZE=$MODEL_SIZE \
    $IMAGE_NAME

echo "Container started successfully!"
echo "Application will be available at: http://localhost:$PORT"