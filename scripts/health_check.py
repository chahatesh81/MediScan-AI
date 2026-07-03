import platform
import sys
from pathlib import Path

import cv2
import fastapi
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

from backend.app.core.config import IMAGE_SIZE, PROJECT_ROOT
from backend.app.core.logging_config import setup_logging
from backend.app.core.reproducibility import set_global_seed


def check(name: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")

    if not condition:
        raise RuntimeError(f"Health check failed: {name}")


def main() -> None:
    print("=" * 60)
    print("MEDISCAN AI — PHASE 0 HEALTH CHECK")
    print("=" * 60)

    set_global_seed()
    logger = setup_logging()

    gpu_devices = tf.config.list_physical_devices("GPU")

    check("Python 3.12", sys.version_info[:2] == (3, 12))
    check("Project root exists", Path(PROJECT_ROOT).exists())
    check("Image size configured", IMAGE_SIZE == (224, 224))
    check("TensorFlow imported", bool(tf.__version__))
    check("RTX GPU detected by TensorFlow", len(gpu_devices) > 0)
    check("NumPy imported", bool(np.__version__))
    check("Pandas imported", bool(pd.__version__))
    check("Scikit-learn imported", bool(sklearn.__version__))
    check("OpenCV imported", bool(cv2.__version__))
    check("FastAPI imported", bool(fastapi.__version__))

    logger.info("Phase 0 health check completed successfully")

    print("\nEnvironment")
    print(f"Python:      {platform.python_version()}")
    print(f"TensorFlow:  {tf.__version__}")
    print(f"GPU:         {gpu_devices}")
    print(f"Project:     {PROJECT_ROOT}")

    print("\n" + "=" * 60)
    print("PHASE 0 STATUS: READY")
    print("=" * 60)


if __name__ == "__main__":
    main()