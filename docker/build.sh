#!/bin/bash

set -a
source docker/.env
set +a

# Only user configuration for build
UID=1000
GID=1000
USER=docker_user
IMAGE_NAME=${IMAGE_NAME:-smolvlm2-app-image}

echo "Building Docker image..."
echo "IMAGE_NAME: $IMAGE_NAME"
echo "USER: $USER"
echo "UID: $UID"
echo "GID: $GID"

docker build -f ./docker/Dockerfile \
             -t $IMAGE_NAME \
             --build-arg UID=$UID \
             --build-arg GID=$GID \
             --build-arg USER=$USER .