"""
Flora Carbon AI - Web Application & Telemetry Dashboard
FastAPI + Interactive Dashboard UI + Classical Spectral Vision Engine
"""

import os
import io
import csv
import json
import base64
import time
import threading
import logging
from typing import Optional, List
import numpy as np
import cv2
import psutil
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import hardware
import analytics
from detector import (
    analyze_tree_canopy, detect_canopy_change, calculate_metrics,
    detect_tree_crowns_multi, compute_deforestation_risk, generate_world_file
)
from report import generate_pdf_report

logger = logging.getLogger("FloraApp")
logging.basicConfig(level=logging.INFO)

SERVER_START_TIME = time.time()

# Real-time in-flight request counter (incremented/decremented around every
# run_in_threadpool call below) - exposed via /api/system/health so concurrency
# is genuinely observed, not asserted.
_concurrency_lock = threading.Lock()
_active_inferences = 0


class _TrackConcurrency:
    def __enter__(self):
        global _active_inferences
        with _concurrency_lock:
            _active_inferences += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _active_inferences
        with _concurrency_lock:
            _active_inferences -= 1

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
                # OSBS/SOAP are real NEON benchmark crops with published, genuine site
                # coordinates. Sundarbans/Amazon/Temperate are procedurally GENERATED
                # illustrative tiles (see create_samples.py) - they are not photos of
                # those real places, so they get no fabricated lat/lon/GSD claim.
                if "OSBS" in f:
                    meta = "NEON OSBS Site • Florida Longleaf Pine Benchmark (real site data)"
                    gsd = 0.10
                    coords = "29°41'47.4\"N 81°59'38.4\"W"
                    lat, lon = 29.6965, -81.9940
                    is_synthetic = False
                elif "SOAP" in f:
                    meta = "NEON SOAP Site • Sierra National Forest Benchmark (real site data)"
                    gsd = 0.10
                    coords = "37°02'01.5\"N 119°15'36.0\"W"
                    lat, lon = 37.0336, -119.2600
                    is_synthetic = False
                elif "Sundarbans" in f:
                    meta = "Synthetic Mangrove-style Ortho (procedurally generated, illustrative only)"
                    gsd = 0.20
                    coords, lat, lon = None, None, None
                    is_synthetic = True
                elif "Amazon" in f:
                    meta = "Synthetic Tropical-Evergreen-style Ortho (procedurally generated, illustrative only)"
                    gsd = 0.15
                    coords, lat, lon = None, None, None
                    is_synthetic = True
                elif "Temperate" in f:
                    meta = "Synthetic Temperate-Conifer-style Ortho (procedurally generated, illustrative only)"
                    gsd = 0.25
                    coords, lat, lon = None, None, None
                    is_synthetic = True
                else:
                    meta = "Synthetic Orthomosaic-style Tile (procedurally generated, illustrative only)"
                    gsd = 0.20
                    coords, lat, lon = None, None, None
                    is_synthetic = True

                if lat is not None:
                    # Real UTM zone/EPSG derived from this sample's actual lat/lon -
                    # never a single fixed value applied regardless of true location.
                    utm_zone = int((lon + 180) / 6) + 1
                    epsg_code = (32600 if lat >= 0 else 32700) + utm_zone
                    hemisphere = "N" if lat >= 0 else "S"
                    epsg = f"EPSG:{epsg_code}"
                    utm_label = f"WGS 84 / UTM Zone {utm_zone}{hemisphere}"
                else:
                    epsg = None
                    utm_label = "Not georeferenced (synthetic tile)"
                    coords = "Not georeferenced (synthetic tile)"

                samples.append({
                    "filename": f,
                    "label": label,
                    "meta": meta,
                    "size_mb": size_mb,
                    "recommended_gsd": gsd,
                    "coords": coords,
                    "lat": lat,
                    "lon": lon,
                    "epsg": epsg,
                    "utm_zone": utm_label,
                    "is_synthetic": is_synthetic,
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


async def _load_kaggle_image_input(dataset_slug: str, filename: str):
    """Resolves a user-picked Kaggle dataset image to a (PIL image, display name) pair,
    reusing the same kagglehub cache as /api/kaggle/images so picking an image the
    user already browsed doesn't re-download anything."""
    _require_kaggle_credentials()
    import kagglehub
    dataset_path = await run_in_threadpool(kagglehub.dataset_download, dataset_slug)
    image_path = await run_in_threadpool(_resolve_kaggle_image_path, dataset_path, filename)
    return Image.open(image_path).convert("RGB"), os.path.basename(image_path)


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
    roi_geojson: Optional[str] = Form(None),
    origin_lat: Optional[float] = Form(None),
    origin_lon: Optional[float] = Form(None),
    kaggle_dataset_slug: Optional[str] = Form(None),
    kaggle_filename: Optional[str] = Form(None)
):
    """
    Main detection analysis endpoint:
    Accepts uploaded files, sample image filenames, OR a (kaggle_dataset_slug,
    kaggle_filename) pair identifying one real image from an already-browsed
    Kaggle dataset - all three run through the exact same detection pipeline
    and return the exact same response shape (annotated image, boxes, telemetry,
    exports). This is what lets a Kaggle image be analyzed and visualized just
    like an upload, not just averaged into aggregate benchmark statistics.

    origin_lat/origin_lon are optional and REAL only when the caller actually knows
    them (e.g. a bundled sample with genuine published site coordinates). When
    omitted - uploads, Kaggle images, or synthetic/illustrative samples - this
    endpoint does NOT fabricate a location: geojson/world-file exports use an
    arbitrary local (0,0) origin and are labeled "not georeferenced" rather than
    silently claiming a real place.
    """
    # Clamp user-supplied sliders to sane bounds regardless of what the client sends.
    confidence = min(0.90, max(0.10, confidence))
    gsd = min(5.0, max(0.01, gsd))
    if heatmap_mode not in ("density", "confidence"):
        heatmap_mode = "density"
    roi_polygon = _parse_roi_polygon(roi_geojson)
    is_georeferenced = origin_lat is not None and origin_lon is not None
    effective_lat = origin_lat if is_georeferenced else 0.0
    effective_lon = origin_lon if is_georeferenced else 0.0

    try:
        if kaggle_dataset_slug and kaggle_filename:
            pil_img, filename = await _load_kaggle_image_input(kaggle_dataset_slug, kaggle_filename)
        else:
            contents = await file.read() if (file is not None and file.filename) else None
            pil_img, filename = _load_image_input(file, contents, sample_filename)
        img_np = np.array(pil_img)

        # Run the CPU-bound detection engine in a worker thread so one heavy request
        # (e.g. a large orthomosaic) doesn't stall the event loop for every other
        # concurrent user - important on a single-process free-tier deployment.
        with _TrackConcurrency():
            result = await run_in_threadpool(
                analyze_tree_canopy,
                image_input=img_np,
                confidence_threshold=confidence,
                gsd_meters_per_pixel=gsd,
                show_boxes=show_boxes,
                show_heatmap=show_heatmap,
                show_centroids=show_centroids,
                heatmap_mode=heatmap_mode,
                roi_polygon=roi_polygon,
                origin_lat=effective_lat,
                origin_lon=effective_lon,
                is_georeferenced=is_georeferenced
            )

        # Encode annotated image to Base64 JPEG/PNG for instant DOM rendering
        annotated_bgr = cv2.cvtColor(result["annotated_image"], cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        m = result["metrics"]

        if is_georeferenced:
            # Real UTM zone/EPSG derived from THIS image's actual known origin -
            # never a single fixed zone applied regardless of true location.
            utm_zone = int((effective_lon + 180) / 6) + 1
            epsg_code = (32600 if effective_lat >= 0 else 32700) + utm_zone
            hemisphere = "N" if effective_lat >= 0 else "S"
            projection_label = f"EPSG:{epsg_code} (WGS 84 / UTM Zone {utm_zone}{hemisphere})"
            world_file_text = generate_world_file(effective_lat, effective_lon, gsd)
        else:
            # No real location known for this image (upload, Kaggle image, or a
            # synthetic/illustrative sample) - say so honestly instead of guessing.
            projection_label = "Not georeferenced (no known location for this image)"
            world_file_text = None

        # Real session logging - powers the analytics dashboard (history, global
        # counters, leaderboard). Never raises into the response on failure.
        analytics.log_analysis(
            filename=filename,
            tree_count=m.get("tree_count", 0),
            canopy_m2=m.get("total_canopy_m2", 0.0),
            cover_pct=m.get("canopy_cover_percentage", 0.0),
            co2e_tons=m.get("inferred_co2e_tons", 0.0),
            inference_time_ms=result["inference_time_ms"],
            source="analyze"
        )

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
            "csv_text": result["csv_text"],
            "world_file_text": world_file_text,
            "projection_label": projection_label,
            "is_georeferenced": is_georeferenced,
            "origin_lat": origin_lat,
            "origin_lon": origin_lon
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

        with _TrackConcurrency():
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

        risk = compute_deforestation_risk(
            cover_pct=result_b["metrics"].get("canopy_cover_percentage", 0.0),
            canopy_loss_m2=change["canopy_loss_m2"],
            canopy_gain_m2=change["canopy_gain_m2"]
        )

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
            "overlay_image_data": f"data:image/jpeg;base64,{overlay_b64}",
            "risk": risk
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
    roi_geojson: Optional[str] = Form(None),
    origin_lat: Optional[float] = Form(None),
    origin_lon: Optional[float] = Form(None),
    kaggle_dataset_slug: Optional[str] = Form(None),
    kaggle_filename: Optional[str] = Form(None)
):
    """
    Re-runs the same analysis as /api/analyze (stateless - no server-side caching)
    and returns a downloadable PDF audit certificate instead of JSON. See
    /api/analyze's docstring re: origin_lat/lon and the Kaggle image source - no
    location is fabricated when the caller doesn't actually know one.
    """
    confidence = min(0.90, max(0.10, confidence))
    gsd = min(5.0, max(0.01, gsd))
    if heatmap_mode not in ("density", "confidence"):
        heatmap_mode = "density"
    roi_polygon = _parse_roi_polygon(roi_geojson)
    is_georeferenced = origin_lat is not None and origin_lon is not None
    effective_lat = origin_lat if is_georeferenced else 0.0
    effective_lon = origin_lon if is_georeferenced else 0.0

    try:
        if kaggle_dataset_slug and kaggle_filename:
            pil_img, filename = await _load_kaggle_image_input(kaggle_dataset_slug, kaggle_filename)
        else:
            contents = await file.read() if (file is not None and file.filename) else None
            pil_img, filename = _load_image_input(file, contents, sample_filename)

        with _TrackConcurrency():
            result = await run_in_threadpool(
                analyze_tree_canopy,
                image_input=np.array(pil_img),
                confidence_threshold=confidence,
                gsd_meters_per_pixel=gsd,
                show_boxes=show_boxes,
                show_heatmap=show_heatmap,
                show_centroids=show_centroids,
                heatmap_mode=heatmap_mode,
                roi_polygon=roi_polygon,
                origin_lat=effective_lat,
                origin_lon=effective_lon,
                is_georeferenced=is_georeferenced
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


@app.post("/api/gsd-sensitivity")
async def api_gsd_sensitivity(
    boxes_json: str = Form(...),
    image_width: int = Form(...),
    image_height: int = Form(...),
    gsd_min: float = Form(0.05),
    gsd_max: float = Form(1.0),
    steps: int = Form(8)
):
    """
    Recomputes canopy area/carbon metrics across a range of GSD values using boxes
    the client already has (from a prior /api/analyze call) - pure math on
    calculate_metrics(), no re-detection, near-zero cost.
    """
    try:
        boxes = json.loads(boxes_json)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="boxes_json must be a valid JSON array")

    steps = max(2, min(20, steps))
    gsd_min = max(0.01, gsd_min)
    gsd_max = max(gsd_min + 0.01, min(5.0, gsd_max))

    curve = []
    for i in range(steps):
        gsd = gsd_min + (gsd_max - gsd_min) * (i / (steps - 1))
        m = calculate_metrics(boxes, (image_height, image_width), gsd)
        curve.append({
            "gsd": round(gsd, 4),
            "total_canopy_m2": m["total_canopy_m2"],
            "inferred_co2e_tons": m["inferred_co2e_tons"]
        })

    return JSONResponse(content={"success": True, "curve": curve})


@app.post("/api/sensitivity")
async def api_sensitivity(
    file: Optional[UploadFile] = File(None),
    sample_filename: Optional[str] = Form(None)
):
    """
    Threshold sensitivity sweep: runs contour extraction once and reports the
    real detected tree count at several confidence thresholds (count determined
    by hardware.py's tier - more thresholds on higher-RAM machines), so the
    curve is always genuinely computed, never simulated.
    """
    try:
        contents = await file.read() if (file is not None and file.filename) else None
        pil_img, filename = _load_image_input(file, contents, sample_filename)
        img_np = np.array(pil_img)

        n_points = hardware.CONFIG["sensitivity_sweep_points"]
        thresholds = [round(0.10 + (0.80 * i / (n_points - 1)), 3) for i in range(n_points)]

        with _TrackConcurrency():
            counts = await run_in_threadpool(detect_tree_crowns_multi, img_np, thresholds)

        curve = [{"threshold": t, "tree_count": counts[t]} for t in thresholds]
        return JSONResponse(content={"success": True, "filename": filename, "curve": curve})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during sensitivity sweep")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch")
async def api_batch(
    files: List[UploadFile] = File(...),
    confidence: float = Form(0.40),
    gsd: float = Form(0.20)
):
    """
    Processes multiple images through the existing detector, sequentially in
    chunks sized by hardware.CONFIG["batch_concurrency"] (1 on low-RAM tiers,
    up to 4 on high-RAM machines) so this never spikes past what a single /api/analyze
    call would use per-image. Returns a per-file summary plus an aggregate CSV.
    """
    confidence = min(0.90, max(0.10, confidence))
    gsd = min(5.0, max(0.01, gsd))
    chunk_size = hardware.CONFIG["batch_concurrency"]

    results = []
    errors = []

    for i in range(0, len(files), chunk_size):
        chunk = files[i:i + chunk_size]
        chunk_inputs = []
        for f in chunk:
            try:
                contents = await f.read()
                if len(contents) > MAX_UPLOAD_BYTES:
                    errors.append({"filename": f.filename, "error": "File exceeds upload limit"})
                    continue
                pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
                chunk_inputs.append((f.filename, np.array(pil_img)))
            except UnidentifiedImageError:
                errors.append({"filename": f.filename, "error": "Not a valid image"})

        with _TrackConcurrency():
            for filename, img_np in chunk_inputs:
                try:
                    result = await run_in_threadpool(
                        analyze_tree_canopy,
                        image_input=img_np,
                        confidence_threshold=confidence,
                        gsd_meters_per_pixel=gsd,
                        show_boxes=False,
                        show_heatmap=False,
                        show_centroids=False
                    )
                    m = result["metrics"]
                    analytics.log_analysis(
                        filename=filename,
                        tree_count=m.get("tree_count", 0),
                        canopy_m2=m.get("total_canopy_m2", 0.0),
                        cover_pct=m.get("canopy_cover_percentage", 0.0),
                        co2e_tons=m.get("inferred_co2e_tons", 0.0),
                        inference_time_ms=result["inference_time_ms"],
                        source="batch"
                    )
                    results.append({"filename": filename, "metrics": m})
                except Exception as e:
                    errors.append({"filename": filename, "error": str(e)})

    aggregate = {
        "total_files": len(files),
        "processed": len(results),
        "failed": len(errors),
        "total_trees": sum(r["metrics"]["tree_count"] for r in results),
        "total_canopy_m2": round(sum(r["metrics"]["total_canopy_m2"] for r in results), 2),
        "total_co2e_tons": round(sum(r["metrics"]["inferred_co2e_tons"] for r in results), 2)
    }

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["filename", "tree_count", "total_canopy_m2", "canopy_cover_percentage", "inferred_co2e_tons"])
    writer.writeheader()
    for r in results:
        writer.writerow({
            "filename": r["filename"],
            "tree_count": r["metrics"]["tree_count"],
            "total_canopy_m2": r["metrics"]["total_canopy_m2"],
            "canopy_cover_percentage": r["metrics"]["canopy_cover_percentage"],
            "inferred_co2e_tons": r["metrics"]["inferred_co2e_tons"]
        })

    return JSONResponse(content={
        "success": True,
        "aggregate": aggregate,
        "results": results,
        "errors": errors,
        "csv_text": csv_buffer.getvalue(),
        "batch_chunk_size": chunk_size
    })


@app.get("/api/system/health")
async def api_system_health():
    """
    Live, real system telemetry - proves the hardware-adaptive/RAM-budget claims
    instead of just asserting them: actual current process RAM, host CPU load,
    uptime, detected hardware tier, and in-flight request count.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    vmem = psutil.virtual_memory()

    return JSONResponse(content={
        "hardware_profile": hardware.get_profile(),
        "process_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
        "system_ram_used_pct": vmem.percent,
        "system_ram_available_gb": round(vmem.available / (1024 ** 3), 2),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
        "active_inferences": _active_inferences
    })


@app.get("/api/analytics/history")
async def api_analytics_history(limit: int = 100):
    return JSONResponse(content={"success": True, "history": analytics.get_history(limit=min(limit, 1000))})


@app.get("/api/analytics/stats")
async def api_analytics_stats():
    return JSONResponse(content={"success": True, "stats": analytics.get_global_stats()})


@app.get("/api/analytics/leaderboard")
async def api_analytics_leaderboard(limit: int = 20):
    return JSONResponse(content={"success": True, "leaderboard": analytics.get_leaderboard(limit=min(limit, 100))})


@app.get("/api/analytics/export")
async def api_analytics_export():
    data = analytics.export_all()
    return Response(
        content=json.dumps(data, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="flora_analytics_export_{int(time.time())}.json"'}
    )


# --- Kaggle dataset integration ---
# A small curated list of real, public Kaggle datasets usable for descriptive
# benchmarking of the canopy detector. Requires the user's own Kaggle API
# credentials (KAGGLE_USERNAME/KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json) -
# every endpoint below degrades to a clear error, never fake data, if those
# aren't configured.
KAGGLE_CURATED_DATASETS = [
    {"slug": "mcagriaksoy/trees-in-satellite-imagery", "label": "Trees in Satellite Imagery"},
]


def _kaggle_credentials_available() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))


def _require_kaggle_credentials():
    if not _kaggle_credentials_available():
        raise HTTPException(
            status_code=503,
            detail="Kaggle credentials not configured. Set KAGGLE_USERNAME/KAGGLE_KEY "
                   "env vars or place a kaggle.json at ~/.kaggle/kaggle.json."
        )


def _list_kaggle_dataset_images(dataset_path: str) -> list:
    image_files = []
    for root, _, files_in_dir in os.walk(dataset_path):
        for fname in files_in_dir:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                image_files.append(os.path.join(root, fname))
    return sorted(image_files)


def _resolve_kaggle_image_path(dataset_path: str, filename: str) -> str:
    """Resolves a user-picked filename to a real file inside the already-downloaded
    dataset directory, rejecting anything that isn't actually inside it (basename-only
    matching + a containment check prevents path traversal via a crafted filename)."""
    safe_name = os.path.basename(filename)
    for candidate in _list_kaggle_dataset_images(dataset_path):
        if os.path.basename(candidate) == safe_name:
            resolved = os.path.realpath(candidate)
            if resolved.startswith(os.path.realpath(dataset_path)):
                return resolved
    raise HTTPException(status_code=404, detail=f"Image '{filename}' not found in this dataset")


@app.get("/api/kaggle/images")
async def api_kaggle_images(dataset_slug: str, limit: int = 24):
    """
    Lists real image files from a downloaded (and kagglehub-cached) Kaggle dataset,
    so the user can pick one to run through the exact same detection pipeline as
    an upload - instead of only seeing aggregate benchmark statistics.
    """
    _require_kaggle_credentials()
    try:
        import kagglehub
        dataset_path = await run_in_threadpool(kagglehub.dataset_download, dataset_slug)
        image_files = await run_in_threadpool(_list_kaggle_dataset_images, dataset_path)
        if not image_files:
            raise HTTPException(status_code=404, detail=f"No images found in dataset '{dataset_slug}'")

        limit = max(1, min(100, limit))
        items = []
        for path in image_files[:limit]:
            fname = os.path.basename(path)
            items.append({
                "filename": fname,
                "size_kb": round(os.path.getsize(path) / 1024, 1),
                "thumbnail_url": f"/api/kaggle/image?dataset_slug={dataset_slug}&filename={fname}"
            })
        return JSONResponse(content={
            "success": True,
            "dataset_slug": dataset_slug,
            "total_images_in_dataset": len(image_files),
            "images": items
        })
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(status_code=500, detail="kagglehub is not installed")
    except Exception as e:
        logger.exception("Error listing Kaggle dataset images")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kaggle/image")
async def api_kaggle_image(dataset_slug: str, filename: str):
    """Serves one real image file from a cached Kaggle dataset (for thumbnails/preview)."""
    _require_kaggle_credentials()
    try:
        import kagglehub
        dataset_path = await run_in_threadpool(kagglehub.dataset_download, dataset_slug)
        image_path = await run_in_threadpool(_resolve_kaggle_image_path, dataset_path, filename)
        ext = os.path.splitext(image_path)[1].lower()
        media_type = "image/png" if ext == ".png" else "image/jpeg"
        with open(image_path, "rb") as f:
            return Response(content=f.read(), media_type=media_type)
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(status_code=500, detail="kagglehub is not installed")
    except Exception as e:
        logger.exception("Error serving Kaggle dataset image")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kaggle/datasets")
async def api_kaggle_datasets():
    if not _kaggle_credentials_available():
        raise HTTPException(
            status_code=503,
            detail="Kaggle credentials not configured. Set KAGGLE_USERNAME/KAGGLE_KEY "
                   "env vars or place a kaggle.json at ~/.kaggle/kaggle.json to enable "
                   "Kaggle-backed benchmarking."
        )
    return JSONResponse(content={"success": True, "datasets": KAGGLE_CURATED_DATASETS})


@app.post("/api/kaggle/benchmark")
async def api_kaggle_benchmark(
    dataset_slug: str = Form(...),
    confidence: float = Form(0.40),
    gsd: float = Form(0.20)
):
    """
    Downloads a real Kaggle dataset (cached locally by kagglehub so repeat runs
    don't re-fetch) and runs the actual detector against a sample of its images,
    bounded by hardware.CONFIG["kaggle_benchmark_sample_size"]. Reports genuine
    descriptive detection statistics computed live - since arbitrary Kaggle
    datasets rarely ship crown-box annotations matching our format, this does
    NOT fabricate precision/recall; a labeled-dataset accuracy mode is tracked
    separately in ROADMAP.md.
    """
    if not _kaggle_credentials_available():
        raise HTTPException(
            status_code=503,
            detail="Kaggle credentials not configured. Set KAGGLE_USERNAME/KAGGLE_KEY "
                   "env vars or place a kaggle.json at ~/.kaggle/kaggle.json."
        )
    confidence = min(0.90, max(0.10, confidence))
    gsd = min(5.0, max(0.01, gsd))

    try:
        import kagglehub

        dataset_path = await run_in_threadpool(kagglehub.dataset_download, dataset_slug)

        image_files = []
        for root, _, files_in_dir in os.walk(dataset_path):
            for fname in files_in_dir:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_files.append(os.path.join(root, fname))

        if not image_files:
            raise HTTPException(status_code=404, detail=f"No images found in dataset '{dataset_slug}'")

        sample_size = hardware.CONFIG["kaggle_benchmark_sample_size"]
        sample = image_files[:sample_size]

        per_image = []
        with _TrackConcurrency():
            for img_path in sample:
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    result = await run_in_threadpool(
                        analyze_tree_canopy,
                        image_input=np.array(pil_img),
                        confidence_threshold=confidence,
                        gsd_meters_per_pixel=gsd,
                        show_boxes=False,
                        show_heatmap=False,
                        show_centroids=False
                    )
                    per_image.append({
                        "filename": os.path.basename(img_path),
                        "tree_count": result["metrics"]["tree_count"],
                        "canopy_cover_percentage": result["metrics"]["canopy_cover_percentage"],
                        "inference_time_ms": result["inference_time_ms"]
                    })
                except Exception as img_err:
                    logger.warning(f"Skipping unreadable benchmark image {img_path}: {img_err}")

        if not per_image:
            raise HTTPException(status_code=500, detail="Could not process any images from the dataset")

        avg_tree_count = round(sum(p["tree_count"] for p in per_image) / len(per_image), 2)
        avg_cover_pct = round(sum(p["canopy_cover_percentage"] for p in per_image) / len(per_image), 2)
        avg_inference_ms = round(sum(p["inference_time_ms"] for p in per_image) / len(per_image), 2)

        return JSONResponse(content={
            "success": True,
            "dataset_slug": dataset_slug,
            "images_sampled": len(per_image),
            "avg_tree_count": avg_tree_count,
            "avg_canopy_cover_percentage": avg_cover_pct,
            "avg_inference_time_ms": avg_inference_ms,
            "per_image": per_image,
            "note": "Descriptive detection statistics from real dataset images - not a "
                    "precision/recall accuracy benchmark, since this dataset doesn't ship "
                    "crown-box ground truth in our format."
        })

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(status_code=500, detail="kagglehub is not installed")
    except Exception as e:
        logger.exception("Error during Kaggle benchmark")
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

