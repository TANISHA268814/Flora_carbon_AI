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
- **Interactive Dashboard UI**: Single FastAPI-served dashboard at `/`, decoupled-frontend-friendly (see `static-frontend/` for a standalone-hosted build).

---

## 🛠️ Project Structure

```
canopy_vision_AI/
├── app.py                 # FastAPI web application + interactive dashboard UI
├── detector.py            # Spectral (Excess Green Index) detection engine, telemetry formulas & OpenCV visualizer
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

---

## ⚠️ Audited Machine Learning Limitations (Failure Modes)

1. **Dense Canopy Overlaps**: Merged interlocking crowns in mature rainforests may be grouped into single bounding boxes by 8–14% without LiDAR height profiling.
2. **Deep Shadow Regions**: Cloud cast and steep terrain shadows reduce spectral reflectance, causing confidence degradation below 0.35 threshold.
3. **Resolution Drop (Low-GSD Drift)**: Imagery with GSD > 0.45 m/pixel lacks sub-meter crown border resolution, leading to false-positive canopy area inflation (~6.2%).
