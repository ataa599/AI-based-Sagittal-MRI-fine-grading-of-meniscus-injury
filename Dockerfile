FROM python:3.10-slim

# Install system libraries required by OpenCV (cv2) and other dependencies
# libgl1: OpenGL library (fixes "libGL.so.1: cannot open shared object file")
# libglib2.0-0: GLib library for low-level utilities
# libsm6, libxext6, libxrender1: X11 extensions for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app
RUN pip install --upgrade pip 
# RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu 
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]