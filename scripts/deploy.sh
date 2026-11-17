#!/bin/bash

echo "Building Docker image..."
docker build -t images-classifier-api .

if [ $? -eq 0 ]; then
    echo "Image built successfully!"
    echo ""
    echo "Starting container..."
    docker run -p 8000:8000 --name images-api images-classifier-api
else
    echo "Build failed!"
    exit 1
fi