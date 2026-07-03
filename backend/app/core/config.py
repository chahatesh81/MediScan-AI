from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
LOG_DIR = PROJECT_ROOT / "logs"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
RANDOM_SEED = 42

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
MODEL_NAME = "mediscan_pneumonia_classifier"
MODEL_VERSION = "1.0.0"