FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install. No ML framework (torch/deepforest) here by design -
# detector.py runs a classical OpenCV-only engine so the whole image/container fits
# comfortably inside a 512MB free-tier RAM limit (e.g. Render's free Web Service tier).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY . .

# Generate sample benchmark images if not present
RUN python create_samples.py

# Expose port (7860 is standard for Hugging Face Spaces;
# platforms like Cloud Run/Render/Railway override this via $PORT at runtime)
EXPOSE 7860
ENV PORT=7860

# Shell form so $PORT is resolved at container start (required for Cloud Run/Render/Railway)
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
