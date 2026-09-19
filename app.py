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

from detector import analyze_tree_canopy, detect_canopy_change
from report import generate_pdf_report

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


def _load_image_input(
    file: Optional[UploadFile],
    contents: Optional[bytes],
    sample_filename: Optional[str]
):
    """Shared resolution of an (uploaded file | sample filename | default sample) -> (PIL image, name)."""
    if file is not None and file.filename:
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit."
            )
        try:
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
        return pil_img, file.filename
    elif sample_filename:
        # Prevent path traversal via a crafted sample_filename (e.g. "../../etc/passwd").
        safe_name = os.path.basename(sample_filename)
        sample_path = os.path.join(SAMPLE_DIR, safe_name)
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=404, detail="Sample image not found")
        return Image.open(sample_path).convert("RGB"), safe_name
    else:
        samples = sorted(
            f for f in os.listdir(SAMPLE_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        )
        if not samples:
            raise HTTPException(status_code=400, detail="No image provided and no samples found")
        sample_path = os.path.join(SAMPLE_DIR, samples[0])
        return Image.open(sample_path).convert("RGB"), samples[0]


def _parse_roi_polygon(roi_geojson: Optional[str]) -> Optional[list]:
    """Parses an optional JSON-encoded list of [x, y] points sent from the ROI canvas tool."""
    if not roi_geojson:
        return None
    try:
        points = json.loads(roi_geojson)
        if isinstance(points, list) and len(points) >= 3:
            return points
    except (ValueError, TypeError):
        pass
    return None


@app.post("/api/analyze")
async def api_analyze(
    file: Optional[UploadFile] = File(None),
    sample_filename: Optional[str] = Form(None),
    confidence: float = Form(0.40),
    gsd: float = Form(0.20),
    show_boxes: bool = Form(True),
    show_heatmap: bool = Form(False),
    show_centroids: bool = Form(True),
    heatmap_mode: str = Form("density"),
    roi_geojson: Optional[str] = Form(None)
):
    """
    Main detection analysis endpoint:
    Accepts uploaded files or sample image filenames, executes spectral crown detection,
    computes carbon/canopy metrics, and returns annotations and telemetry data.
    """
    # Clamp user-supplied sliders to sane bounds regardless of what the client sends.
    confidence = min(0.90, max(0.10, confidence))
    gsd = min(5.0, max(0.01, gsd))
    if heatmap_mode not in ("density", "confidence"):
        heatmap_mode = "density"
    roi_polygon = _parse_roi_polygon(roi_geojson)

    try:
        contents = await file.read() if (file is not None and file.filename) else None
        pil_img, filename = _load_image_input(file, contents, sample_filename)
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
            show_centroids=show_centroids,
            heatmap_mode=heatmap_mode,
            roi_polygon=roi_polygon
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


def _encode_jpeg_b64(image_rgb: np.ndarray, quality: int = 92) -> str:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/api/compare")
async def api_compare(
    file_a: Optional[UploadFile] = File(None),
    file_b: Optional[UploadFile] = File(None),
    sample_a: Optional[str] = Form(None),
    sample_b: Optional[str] = Form(None),
    confidence: float = Form(0.40),
    gsd: float = Form(0.20)
):
    """
    Change-detection endpoint: runs the core detector on two images of the same site
    (e.g. two survey dates) and reports canopy loss/gain between them. No
    georectification is performed - image B is aligned to image A by a plain resize,
    so this assumes both tiles already frame the same extent.
    """
    confidence = min(0.90, max(0.10, confidence))
    gsd = min(5.0, max(0.01, gsd))

    try:
        contents_a = await file_a.read() if (file_a is not None and file_a.filename) else None
        contents_b = await file_b.read() if (file_b is not None and file_b.filename) else None
        pil_a, name_a = _load_image_input(file_a, contents_a, sample_a)
        pil_b, name_b = _load_image_input(file_b, contents_b, sample_b)

        result_a = await run_in_threadpool(
            analyze_tree_canopy,
            image_input=np.array(pil_a),
            confidence_threshold=confidence,
            gsd_meters_per_pixel=gsd,
            show_boxes=True,
            show_heatmap=False,
            show_centroids=False
        )
        result_b = await run_in_threadpool(
            analyze_tree_canopy,
            image_input=np.array(pil_b),
            confidence_threshold=confidence,
            gsd_meters_per_pixel=gsd,
            show_boxes=True,
            show_heatmap=False,
            show_centroids=False
        )

        change = await run_in_threadpool(detect_canopy_change, result_a, result_b, gsd)

        overlay_b64 = _encode_jpeg_b64(cv2.addWeighted(
            result_a["annotated_image"], 0.55, change["overlay_mask"], 0.45, 0
        ))

        return JSONResponse(content={
            "success": True,
            "filename_a": name_a,
            "filename_b": name_b,
            "metrics_a": result_a["metrics"],
            "metrics_b": result_b["metrics"],
            "canopy_loss_m2": change["canopy_loss_m2"],
            "canopy_gain_m2": change["canopy_gain_m2"],
            "net_change_m2": change["net_change_m2"],
            "net_change_co2e_tons": change["net_change_co2e_tons"],
            "overlay_image_data": f"data:image/jpeg;base64,{overlay_b64}"
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during change detection")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/report")
async def api_report(
    file: Optional[UploadFile] = File(None),
    sample_filename: Optional[str] = Form(None),
    confidence: float = Form(0.40),
    gsd: float = Form(0.20),
    show_boxes: bool = Form(True),
    show_heatmap: bool = Form(False),
    show_centroids: bool = Form(True),
    heatmap_mode: str = Form("density"),
    roi_geojson: Optional[str] = Form(None)
):
    """
    Re-runs the same analysis as /api/analyze (stateless - no server-side caching)
    and returns a downloadable PDF audit certificate instead of JSON.
    """
    confidence = min(0.90, max(0.10, confidence))
    gsd = min(5.0, max(0.01, gsd))
    if heatmap_mode not in ("density", "confidence"):
        heatmap_mode = "density"
    roi_polygon = _parse_roi_polygon(roi_geojson)

    try:
        contents = await file.read() if (file is not None and file.filename) else None
        pil_img, filename = _load_image_input(file, contents, sample_filename)

        result = await run_in_threadpool(
            analyze_tree_canopy,
            image_input=np.array(pil_img),
            confidence_threshold=confidence,
            gsd_meters_per_pixel=gsd,
            show_boxes=show_boxes,
            show_heatmap=show_heatmap,
            show_centroids=show_centroids,
            heatmap_mode=heatmap_mode,
            roi_polygon=roi_polygon
        )

        annotated_bgr = cv2.cvtColor(result["annotated_image"], cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

        pdf_bytes = await run_in_threadpool(
            generate_pdf_report, result, filename, buffer.tobytes()
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="flora_carbon_report_{int(time.time())}.pdf"'}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating report")
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

