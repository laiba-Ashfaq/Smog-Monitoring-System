#!/usr/bin/env bash
# ==============================================================================
# Alibaba Cloud ECS Deployment Script for Smog Sentinel Punjab
# ==============================================================================

set -e

echo "🚀 Starting Deployment of Smog Sentinel Punjab on Alibaba Cloud ECS..."

# Step 1: Update System Packages & Install Docker
sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose

# Step 2: Build Docker Image
echo "📦 Building Docker Container Image..."
docker build -t smog-sentinel-punjab:latest -f deployment/Dockerfile .

# Step 3: Run Container on Port 8501
echo "🌐 Launching Application Container..."
docker stop smog-sentinel-app || true
docker rm smog-sentinel-app || true

docker run -d \
  --name smog-sentinel-app \
  --restart always \
  -p 8501:8501 \
  smog-sentinel-punjab:latest

echo "✅ Deployment Complete! Access the dashboard at: http://<ALIBABA_ECS_PUBLIC_IP>:8501"
