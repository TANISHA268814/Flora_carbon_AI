"""
Flora Carbon AI - Web Application & Telemetry Dashboard
FastAPI + Interactive Dashboard UI + Classical Spectral Vision Engine
"""

import os
import io
import json
import base64
import time
import logging
from typing import Optional, List
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from detector import analyze_tree_canopy

logger = logging.getLogger("FloraApp")
logging.basicConfig(level=logging.INFO)

# Root Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Max accepted upload size. Free-tier hosts have limited RAM; a very large raw file
# (e.g. an uncompressed multi-band GeoTIFF) can OOM the container during decode.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# Initialize FastAPI App
app = FastAPI(
    title="Flora Carbon AI",
    description="Autonomous Tree Crown Detection & Canopy Carbon Telemetry",
    version="1.4.0"
)

# Allow the separately-hosted static frontend (a different origin, e.g. a Hugging Face
# Static Space) to call this API. No cookies/auth are used, so a wildcard origin is safe.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount sample images
app.mount("/sample_images", StaticFiles(directory=SAMPLE_DIR), name="sample_images")


@app.get("/health")
async def health_check():
    """Lightweight liveness probe for the hosting platform - never blocks on model load."""
    return {"status": "ok"}


@app.get("/api/samples")
async def list_sample_images():
    """Returns available sample orthomosaic tiles for one-click testing."""
    samples = []
    if os.path.exists(SAMPLE_DIR):
        for f in sorted(os.listdir(SAMPLE_DIR)):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                file_path = os.path.join(SAMPLE_DIR, f)
                size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                
                label = f.replace("_", " ").replace(".png", "").replace(".jpg", "")
                if "Sundarbans" in f:
                    meta = "Sundarbans Biosphere - Sector 4B • Mangrove Ortho"
                    gsd = 0.20
                    coords = "21°56'58.9\"N 88°53'58.9\"E"
                elif "Amazon" in f:
                    meta = "Amazon Basin - Acre Sector 12 • Tropical Evergreen"
                    gsd = 0.15
                    coords = "9°58'29.1\"S 67°48'36.4\"W"
                elif "Temperate" in f:
                    meta = "Pacific Northwest - Plot 7 • Temperate Conifer"
                    gsd = 0.25
                    coords = "45°22'11.3\"N 121°44'02.1\"W"
                elif "OSBS" in f:
                    meta = "NEON OSBS Site • Florida Longleaf Pine Benchmark"
                    gsd = 0.10
                    coords = "29°41'47.4\"N 81°59'38.4\"W"
                elif "SOAP" in f:
                    meta = "NEON SOAP Site • Sierra National Forest Benchmark"
                    gsd = 0.10
                    coords = "37°02'01.5\"N 119°15'36.0\"W"
                else:
                    meta = "UAV Orthomosaic Forest Tile"
                    gsd = 0.20
                    coords = "21.9497° N, 88.8997° E"

                samples.append({
                    "filename": f,
                    "label": label,
                    "meta": meta,
                    "size_mb": size_mb,
                    "recommended_gsd": gsd,
                    "coords": coords,
                    "url": f"/sample_images/{f}"
                })
    return {"samples": samples}


@app.post("/api/analyze")
async def api_analyze(
    file: Optional[UploadFile] = File(None),
    sample_filename: Optional[str] = Form(None),
    confidence: float = Form(0.40),
    gsd: float = Form(0.20),
    show_boxes: bool = Form(True),
    show_heatmap: bool = Form(False),
    show_centroids: bool = Form(True)
):
    """
    Main detection analysis endpoint:
    Accepts uploaded files or sample image filenames, executes spectral crown detection,
    computes carbon/canopy metrics, and returns annotations and telemetry data.
    """
    # Clamp user-supplied sliders to sane bounds regardless of what the client sends.
    confidence = min(0.90, max(0.10, confidence))
    gsd = min(5.0, max(0.01, gsd))

    try:
        if file is not None and file.filename:
            contents = await file.read()
            if len(contents) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit."
                )
            try:
                pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            except UnidentifiedImageError:
                raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
            filename = file.filename
        elif sample_filename:
            # Prevent path traversal via a crafted sample_filename (e.g. "../../etc/passwd").
            safe_name = os.path.basename(sample_filename)
            sample_path = os.path.join(SAMPLE_DIR, safe_name)
            if not os.path.exists(sample_path):
                raise HTTPException(status_code=404, detail="Sample image not found")
            pil_img = Image.open(sample_path).convert("RGB")
            filename = safe_name
        else:
            # Fallback to first available sample
            samples = sorted(
                f for f in os.listdir(SAMPLE_DIR)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
            )
            if not samples:
                raise HTTPException(status_code=400, detail="No image provided and no samples found")
            sample_path = os.path.join(SAMPLE_DIR, samples[0])
            pil_img = Image.open(sample_path).convert("RGB")
            filename = samples[0]

        img_np = np.array(pil_img)

        # Run the CPU-bound detection engine in a worker thread so one heavy request
        # (e.g. a large orthomosaic) doesn't stall the event loop for every other
        # concurrent user - important on a single-process free-tier deployment.
        result = await run_in_threadpool(
            analyze_tree_canopy,
            image_input=img_np,
            confidence_threshold=confidence,
            gsd_meters_per_pixel=gsd,
            show_boxes=show_boxes,
            show_heatmap=show_heatmap,
            show_centroids=show_centroids
        )

        # Encode annotated image to Base64 JPEG/PNG for instant DOM rendering
        annotated_bgr = cv2.cvtColor(result["annotated_image"], cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        response_payload = {
            "success": True,
            "filename": filename,
            "model_source": result["model_source"],
            "inference_time_ms": result["inference_time_ms"],
            "dimensions": result["image_dimensions"],
            "metrics": result["metrics"],
            "limitations": result["limitations"],
            "boxes_count": len(result["boxes"]),
            "boxes": result["boxes"][:150], # Return top boxes for HUD
            "annotated_image_data": f"data:image/jpeg;base64,{img_b64}",
            "geojson": result["geojson"],
            "csv_text": result["csv_text"]
        }
        return JSONResponse(content=response_payload)

    except HTTPException:
        raise  # preserve intended status codes (400/404/413) instead of flattening to 500
    except Exception as e:
        logger.exception("Error during analysis")
        raise HTTPException(status_code=500, detail=str(e))


# Full Stitch Interactive Dashboard Endpoint
@app.get("/", response_class=HTMLResponse)
async def serve_stitch_ui():
    """
    Renders the exact Google Stitch UI dashboard with real-time interactive JavaScript.
    """
    html_file_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(html_file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Starting Flora Carbon AI on http://127.0.0.1:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

