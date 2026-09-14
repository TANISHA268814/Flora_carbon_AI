FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first to keep image lightweight (< 200MB instead of 2.5GB CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY . .

# Generate sample benchmark images if not present
RUN python create_samples.py

# Expose port (7860 is standard for Hugging Face Spaces & Gradio;
# platforms like Cloud Run/Render/Railway override this via $PORT at runtime)
EXPOSE 7860
ENV PORT=7860

# Shell form so $PORT is resolved at container start (required for Cloud Run/Render/Railway)
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
