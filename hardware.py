"""
Flora Carbon AI - Local hardware-adaptive configuration.

Detects the host machine's RAM/CPU once at import time and derives resource
limits from it, so the same codebase self-tunes whether it's run on a 4GB
laptop or a 64GB workstation - less RAM means more aggressive downscaling and
tighter concurrency/history limits; more RAM relaxes them. This runs locally
first (per product decision - cloud deployment is a later, separate concern)
but works identically in any environment since it only reads the live host.
"""

import os
import logging

logger = logging.getLogger("FloraHardware")

try:
    import psutil
    TOTAL_RAM_GB = round(psutil.virtual_memory().total / (1024 ** 3), 2)
except ImportError:
    # Extremely defensive fallback - psutil is a required dependency, but if
    # it's ever missing, assume the smallest tier rather than crashing the app.
    TOTAL_RAM_GB = 4.0

CPU_COUNT = os.cpu_count() or 2

if TOTAL_RAM_GB <= 4.0:
    TIER = "ultra_light"
elif TOTAL_RAM_GB <= 8.0:
    TIER = "light"
elif TOTAL_RAM_GB <= 16.0:
    TIER = "balanced"
else:
    TIER = "performance"

_TIER_CONFIG = {
    "ultra_light": {
        "max_inference_dim": 900,
        "batch_concurrency": 1,
        "analytics_history_limit": 200,
        "sensitivity_sweep_points": 3,
        "kaggle_benchmark_sample_size": 5,
    },
    "light": {
        "max_inference_dim": 1200,
        "batch_concurrency": 1,
        "analytics_history_limit": 500,
        "sensitivity_sweep_points": 4,
        "kaggle_benchmark_sample_size": 10,
    },
    "balanced": {
        "max_inference_dim": 1600,
        "batch_concurrency": 2,
        "analytics_history_limit": 2000,
        "sensitivity_sweep_points": 5,
        "kaggle_benchmark_sample_size": 20,
    },
    "performance": {
        "max_inference_dim": 2200,
        "batch_concurrency": 4,
        "analytics_history_limit": 5000,
        "sensitivity_sweep_points": 6,
        "kaggle_benchmark_sample_size": 40,
    },
}

CONFIG = _TIER_CONFIG[TIER]

logger.info(
    f"Hardware profile: {TOTAL_RAM_GB}GB RAM, {CPU_COUNT} CPUs -> tier '{TIER}' "
    f"(max_inference_dim={CONFIG['max_inference_dim']}, "
    f"batch_concurrency={CONFIG['batch_concurrency']})"
)


def get_profile() -> dict:
    """Snapshot of the detected hardware profile, for the /api/system/health endpoint."""
    return {
        "total_ram_gb": TOTAL_RAM_GB,
        "cpu_count": CPU_COUNT,
        "tier": TIER,
        "config": CONFIG,
    }
