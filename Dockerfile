FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install. requirements.txt itself pins torch/torchvision to the
# CPU-only wheel index via --extra-index-url (NOT --index-url, which would replace PyPI
# entirely and break resolution of transitive deps like typing-extensions/flit_core).
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
