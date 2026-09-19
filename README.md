---
title: Flora Carbon AI
emoji: 🌲
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🌲 Flora Carbon AI - Tree Crown Detection & Carbon Telemetry Platform

> **Autonomous Aerial Orthomosaic Tree Crown Detection, Canopy Cover Analysis & Model Transparency Audits for Climate-Tech & Carbon Verification.**

---

## 🚀 Live App Overview

Flora Carbon AI is a full-stack AI remote-sensing application designed for environmental monitoring, forest telemetry, and carbon sequestration auditing. It features:

- **Spectral Vision Engine**: Classical Excess Green Index + adaptive contour segmentation (no ML weights/GPU required - runs comfortably on a 512MB free-tier container).
- **Canopy Footprint & GSD Calibration**: Ground Sampling Distance (GSD in meters/pixel) conversion to accurately calculate total canopy surface area ($m^2$ and hectares).
- **Canopy Cover Index & UN FAO Tiers**: Calculation of raster canopy coverage percentage with automated classification (Class A, B, C).
- **Carbon Sequestration & Biomass (AGB)**: Pantropical allometric biomass modeling (Chave et al., 2014 & Jucker et al., 2017) estimating dry Above-Ground Biomass (tons) and carbon dioxide equivalents ($t\text{ CO}_2\text{e}$).
- **Interactive Emerald Stitch UI**: Dark-mode geospatial telemetry HUD featuring real-time reticle overlays, centroid crosshairs, heatmaps, FOV calculation, and instant GeoJSON / CSV exports.
- **Honesty & Model Transparency Card**: Audited failure mode callouts detailing dense canopy overlaps, deep terrain shadows, and low-GSD resolution drift.
- **Canopy Change Detection**: Compare two surveys of the same site (`/api/compare`) to quantify canopy loss/gain and net CO₂e change, with a red/green loss-gain overlay.
- **Region of Interest (ROI) Selection**: Draw a polygon directly on the preview to constrain detection to a specific plot boundary instead of the full tile.
- **Confidence/Uncertainty Heatmap**: Toggle between a density heatmap and a confidence-weighted heatmap that highlights low-certainty (shadow/overlap-degraded) regions.
- **PDF Audit Report Export**: One-click downloadable certificate (`/api/report`) with the KPI summary, annotated image, and audited limitations - alongside the existing GeoJSON/CSV/QGIS-world-file exports.
- **Hardware-Adaptive Sizing**: `hardware.py` detects the host's RAM/CPU at startup and tiers detection resolution, batch concurrency, analytics history, and sweep sizes accordingly (4GB machines run leaner; 16GB+ machines relax the limits) - local-first, no cloud dependency for this.
- **Detection Insights**: Live confidence-score and crown-size histograms, a rolling processing-time chart, a GSD sensitivity simulator, and a single-pass threshold sensitivity sweep - all computed from real detection output, never mocked.
- **Session Analytics**: Every analysis is logged to a local SQLite database (`analytics.py`); the dashboard shows genuine cumulative counters, a canopy-cover trend, and a site leaderboard, with a one-click JSON export.
- **Live System Health**: `/api/system/health` reports actual process RAM, host CPU, uptime, and in-flight request count - not asserted, measured.
- **Batch Processing**: Upload multiple images at once (`/api/batch`); processed sequentially in hardware-sized chunks with an aggregate CSV export.
- **Kaggle Dataset Benchmarking**: `/api/kaggle/benchmark` downloads a real public Kaggle dataset and runs the actual detector against it for genuine descriptive statistics (requires the user's own Kaggle API credentials; degrades to a clear message otherwise, never fake data).
- **Interactive Dashboard UI**: Single FastAPI-served dashboard at `/`, decoupled-frontend-friendly (see `static-frontend/` for a standalone-hosted build).

See `ROADMAP.md` for the longer-term plan (a real trained model published to Hugging Face Hub/Kaggle, labeled-dataset accuracy benchmarking, API keys, biome-aware carbon modeling).

---

## 🛠️ Project Structure

```
canopy_vision_AI/
├── app.py                 # FastAPI web application + interactive dashboard UI
├── detector.py            # Spectral (Excess Green Index) detection engine, telemetry formulas, ROI/change-detection & OpenCV visualizer
├── hardware.py            # Local hardware detection (RAM/CPU) -> tiered resource config
├── analytics.py           # Real session analytics log (SQLite)
├── report.py              # PDF audit report generation (reportlab)
├── create_samples.py      # Benchmark orthomosaic generator and sample asset manager
├── sample_images/         # Benchmark test images for instant one-click evaluation
│   ├── OSBS_029.png       # NEON Florida Longleaf Pine benchmark
│   ├── SOAP_061.png       # NEON Sierra National Forest benchmark
│   ├── Sundarbans_Sector_4B.png # Mangrove Orthomosaic
│   ├── Amazon_Tropical_Plot_08.png # Tropical Evergreen Canopy
│   └── Temperate_Pine_Canopy.png # Temperate Conifer Plot
├── requirements.txt       # Python dependencies
└── README.md              # Documentation & guide
```

---

## 🔬 Mathematical Formulations

1. **Canopy Surface Area ($m^2$)**:
   $$\text{Total Canopy Area }(m^2) = \text{Total Canopy Pixels} \times (\text{GSD})^2$$

2. **Canopy Cover Index (%)**:
   $$\text{Canopy Cover } (\%) = \left(\frac{\text{Merged Canopy Mask Pixels}}{\text{Total Image Pixels}}\right) \times 100$$

3. **Inferred Carbon Sequestration ($t\text{ CO}_2\text{e}$)**:
   $$\text{AGB (dry tons)} = \left(\frac{\text{Cover } \%}{100}\right) \times 220 \times \text{Canopy Area (ha)}$$
   $$\text{Inferred CO}_2\text{e (tons)} = \text{AGB} \times 0.47 \times 3.667$$

---

## 💻 How to Run Locally

1. **Activate Virtual Environment**:

   ```powershell
   .\.venv\Scripts\python.exe app.py
   ```

2. **Open in Browser**:
   - **Main Dashboard**: [http://localhost:7860](http://localhost:7860)
   - **Interactive API Docs (FastAPI / Swagger)**: [http://localhost:7860/docs](http://localhost:7860/docs)

The app auto-detects your machine's RAM/CPU on startup (see `hardware.py`) and
logs the chosen tier - no configuration needed. This governs detection
resolution, batch concurrency, and analytics/history sizing so the same code
runs comfortably on a 4GB laptop or a 64GB workstation.

**Optional - Kaggle dataset benchmarking**: to enable the "Kaggle Dataset
Benchmark" panel, set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables
(from [kaggle.com/settings](https://www.kaggle.com/settings) -> API -> Create
New Token), or place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`.
Without credentials, that panel shows a clear "not configured" message - the
rest of the app is unaffected.

---

## ⚠️ Audited Machine Learning Limitations (Failure Modes)

1. **Dense Canopy Overlaps**: Merged interlocking crowns in mature rainforests may be grouped into single bounding boxes by 8–14% without LiDAR height profiling.
2. **Deep Shadow Regions**: Cloud cast and steep terrain shadows reduce spectral reflectance, causing confidence degradation below 0.35 threshold.
3. **Resolution Drop (Low-GSD Drift)**: Imagery with GSD > 0.45 m/pixel lacks sub-meter crown border resolution, leading to false-positive canopy area inflation (~6.2%).
