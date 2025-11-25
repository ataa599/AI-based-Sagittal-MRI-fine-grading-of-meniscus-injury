# Docker Setup for Meniscus Injury Grading Application

This application is containerized with GPU support for PyTorch-based inference.

## Prerequisites

1. **Docker**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) or Docker Engine (Linux)
2. **NVIDIA Docker Runtime**: Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
3. **NVIDIA GPU** with CUDA support
4. **NVIDIA Drivers**: Ensure you have compatible NVIDIA drivers installed (CUDA 11.8 compatible)

### Verify GPU Access

```powershell
# Check if NVIDIA Docker runtime is installed
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Building the Docker Image

### Option 1: Using Docker Compose (Recommended)

```powershell
# Build and start the application
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# Stop the application
docker-compose down
```

### Option 2: Using Docker CLI

```powershell
# Build the image
docker build -t meniscus-app:latest .

# Run the container with GPU support
docker run --gpus all -p 8000:8000 `
  -v ${PWD}/Artifacts:/app/Artifacts `
  -v ${PWD}/logs:/app/logs `
  -v ${PWD}/best_f1_model.pth:/app/best_f1_model.pth `
  -v ${PWD}/best.pt:/app/best.pt `
  --name meniscus-container `
  meniscus-app:latest
```

## Accessing the Application

Once the container is running, access the application at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## API Endpoints

- `GET /` - Home page
- `GET /results` - Results page
- `POST /infer-folder/` - Upload a ZIP file containing DICOM images for inference
- `GET /train/` - Trigger model training pipeline

## Container Management

```powershell
# View running containers
docker ps

# View logs
docker-compose logs -f

# Stop the container
docker-compose down

# Restart the container
docker-compose restart

# Access container shell
docker exec -it meniscus-inference-app bash

# Check GPU inside container
docker exec -it meniscus-inference-app nvidia-smi
```

## Volume Mounts

The following directories are mounted as volumes:
- `./Artifacts` - Training artifacts and datasets
- `./logs` - Application logs
- Model weights: `best_f1_model.pth`, `best.pt`, `best_acc_model.pth`

## Environment Variables

- `CUDA_VISIBLE_DEVICES=0` - Specifies which GPU to use
- `PYTHONUNBUFFERED=1` - Ensures Python output is sent directly to logs

## Troubleshooting

### GPU Not Detected

```powershell
# Check if GPU is accessible
docker run --rm --gpus all meniscus-app:latest nvidia-smi

# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Port Already in Use

```powershell
# Change port in docker-compose.yml
ports:
  - "8080:8000"  # Map to different host port
```

### Out of Memory

Adjust GPU memory limits in docker-compose.yml:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Production Deployment

For production, consider:
1. Using a reverse proxy (nginx)
2. Setting up SSL/TLS certificates
3. Implementing proper authentication
4. Setting resource limits
5. Using Docker secrets for sensitive data

## Updating the Application

```powershell
# Rebuild after code changes
docker-compose up --build -d

# Or force rebuild
docker-compose build --no-cache
docker-compose up -d
```

## Cleaning Up

```powershell
# Remove containers and networks
docker-compose down

# Remove containers, networks, and volumes
docker-compose down -v

# Remove images
docker rmi meniscus-app:latest

# Clean up all unused Docker resources
docker system prune -a
```
