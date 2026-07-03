import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from backend.app.core.config import PROCESSED_DATA_DIR
from backend.app.services.data_pipeline import CLASS_TO_LABEL


MANIFEST_FILE = PROCESSED_DATA_DIR / "final_manifest.csv"


def calculate_class_weights() -> dict[int, float]:
    df = pd.read_csv(MANIFEST_FILE)

    train_df = df[
        df["final_split"] == "train"
    ].copy()

    y = (
        train_df["class_name"]
        .map(CLASS_TO_LABEL)
        .to_numpy(dtype=np.int64)
    )

    classes = np.array(
        sorted(CLASS_TO_LABEL.values()),
        dtype=np.int64,
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )

    return {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }