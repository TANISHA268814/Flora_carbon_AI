# Flora Carbon AI — Roadmap

This tracks the longer-term feature plan beyond the current release.

**Shipped:**
- Phase 1: change detection, PDF reports, ROI selection, confidence heatmap.
- Hardware-adaptive config (`hardware.py`) that tiers resource limits by
  detected host RAM (4/8/16/64GB+), used across detection resolution, batch
  concurrency, analytics history size, sensitivity sweep points, and Kaggle
  benchmark sample size.
- Real session analytics (`analytics.py`, SQLite) - history, global stats,
  leaderboard, JSON export - replacing every hardcoded placeholder number
  that was previously in the dashboard template.
- Detection insights (score/crown-size histograms, processing-time trend, GSD
  sensitivity simulator, threshold sensitivity sweep, QGIS `.pgw` export,
  transparent deforestation risk score).
- Batch processing (`/api/batch`), live system health (`/api/system/health`),
  and Kaggle dataset benchmarking (`/api/kaggle/*`) with graceful degradation
  when Kaggle credentials aren't configured.

Phases below are scoped for follow-up sessions.

## Follow-up: labeled-dataset accuracy benchmark

`/api/kaggle/benchmark` currently reports real *descriptive* detection
statistics (tree count, cover %, inference time) from actual downloaded
images - it deliberately does NOT report precision/recall/IoU, because
arbitrary public Kaggle datasets don't ship crown-bounding-box ground truth in
our format. A true accuracy benchmark needs a specific annotated dataset (e.g.
NEON crown-delineation benchmarks, or a COCO/Pascal-VOC-style tree dataset)
plus a matching-and-scoring step (IoU-based box matching against ground
truth). Scope this separately once a specific annotated source is chosen -
don't retrofit fake precision numbers onto unlabeled datasets.

## Phase 2 — Real trained model, published as an artifact

The current engine is classical computer vision (Excess Green Index + contour
segmentation in `detector.py`) — no trained weights, which is why it fits in
512MB RAM. To have a genuine "model" to publish on Hugging Face Hub / Kaggle:

1. Train or fine-tune a lightweight crown-segmentation model (candidates: a
   small U-Net, or a re-trained DeepForest checkpoint) on NEON + Sundarbans-style
   labeled data. Needs a decision on dataset + training compute (local GPU vs.
   Colab/Kaggle notebooks) before starting.
2. Export to **ONNX** and serve with `onnxruntime` instead of full PyTorch —
   this is the key trick to get real ML accuracy without repeating the
   torch/DeepForest OOM crash-loop that forced their removal in the first
   place (see git history: `ae91d69`). Re-measure RAM footprint before wiring
   it into `app.py`; if it still doesn't fit Render's free tier, it ships as an
   opt-in higher-RAM deployment target, not the default service.
3. Publish the ONNX weights + a model card to Hugging Face Hub. This is the
   actual "model" artifact — the web app stays a separate consumer of it.
4. Publish a Kaggle notebook that loads those weights against a public
   benchmark dataset (NEON/OSBS, etc.) for citation/credibility purposes,
   independent of the web app's own marketing.

## Phase 3 — Platform features

- **Public REST API with API keys**: simple header-based key check
  (`X-API-Key`) validated against a small store (SQLite or a hashed key file
  to start — no paid infra required), plus basic in-memory rate limiting per
  key. Lets drone-ops/GIS tools integrate directly instead of going through
  the dashboard UI.

## Phase 4 — Bigger bets

- **Biome/species classification**: cluster crown color signatures (e.g.
  k-means on hue/saturation stats per detected crown) to pick a per-biome AGB
  density constant instead of the current fixed 220 t/ha figure used in
  `calculate_metrics()` — should measurably improve carbon estimate accuracy
  across very different biomes (mangrove vs. conifer vs. tropical).
- **Historical trend dashboard**: track repeat surveys of the same site over
  time. Needs persistent storage — a small SQLite file is enough for a
  single-instance Render deployment; would need an external DB if the
  service ever scales to multiple instances.
- **Carbon registry methodology alignment**: write up how the AGB/CO2e
  formulas relate to Verra/Gold Standard methodologies, and cite the specific
  allometric equations used (Chave et al. 2014, Jucker et al. 2017 — already
  referenced in `detector.py` comments) so outputs are defensible for real
  carbon-credit conversations, not just demo numbers.

## Branding / trademark (not a code task)

Actual trademark registration is a legal filing, not something done in this
repo. Open decision for the user:

- "Flora Carbon AI" is a fairly descriptive/generic name, which is harder to
  register and defend than a coined name — worth considering a more
  distinctive name if trademark protection matters.
- In the meantime: keep branding (name, colors, logo mark) consistent across
  the README, the dashboard UI, and the HF Space listing, and add a proper
  `LICENSE` file to the repo to establish clear usage terms regardless of
  trademark status.
