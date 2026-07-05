from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf

from backend.app.core.config import (
    IMAGE_SIZE,
    PROJECT_ROOT,
)
from backend.app.ml.xray_preprocessing import (
    preprocess_xray,
)


SELECTION_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "final_model_selection.json"
)

V3_ADDENDUM_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "final_model_selection_v3_addendum.json"
)


class InferenceService:
    """
    Production inference service for MediScan AI.

    Frozen deployment policy:
    - V1 is the authoritative primary classifier.
    - V3 is an exploratory secondary safety signal.
    - V3 never overrides V1.
    - V1/V3 probabilities are never ensembled.
    """

    def __init__(self) -> None:
        self._load_lock = threading.Lock()

        self._v1_model: tf.keras.Model | None = None
        self._v3_model: tf.keras.Model | None = None

        self._load_frozen_configuration()

    def _load_frozen_configuration(self) -> None:
        if not SELECTION_FILE.is_file():
            raise FileNotFoundError(
                f"Missing model selection record: "
                f"{SELECTION_FILE}"
            )

        if not V3_ADDENDUM_FILE.is_file():
            raise FileNotFoundError(
                f"Missing V3 addendum: "
                f"{V3_ADDENDUM_FILE}"
            )

        selection = json.loads(
            SELECTION_FILE.read_text()
        )

        addendum = json.loads(
            V3_ADDENDUM_FILE.read_text()
        )

        primary = selection[
            "selected_primary_model"
        ]

        deployment_policy = addendum[
            "deployment_policy"
        ]

        if primary["name"] != "baseline_cnn_v1":
            raise RuntimeError(
                "Frozen primary model is not "
                "baseline_cnn_v1."
            )

        if (
            deployment_policy[
                "primary_prediction_source"
            ]
            != "baseline_cnn_v1"
        ):
            raise RuntimeError(
                "Unexpected primary prediction source."
            )

        if deployment_policy[
            "automatic_override_allowed"
        ]:
            raise RuntimeError(
                "Automatic V3 override must remain disabled."
            )

        if deployment_policy[
            "automatic_ensemble_allowed"
        ]:
            raise RuntimeError(
                "Automatic ensemble must remain disabled."
            )

        self.v1_model_path = Path(
            primary["model_file"]
        )

        self.v1_threshold = float(
            primary["decision_threshold"]
        )

        v3_record = addendum["advanced_v3"]

        self.v3_model_path = Path(
            v3_record["checkpoint"]
        )

        self.v3_threshold = float(
            v3_record["threshold"]
        )

        self.warning_condition = (
            deployment_policy[
                "recommended_warning_condition"
            ]
        )

        if (
            self.warning_condition
            != "v1_predicts_normal_and_v3_predicts_pneumonia"
        ):
            raise RuntimeError(
                "Unexpected V3 warning condition."
            )

        if not self.v1_model_path.is_file():
            raise FileNotFoundError(
                f"Missing V1 model: "
                f"{self.v1_model_path}"
            )

        if not self.v3_model_path.is_file():
            raise FileNotFoundError(
                f"Missing V3 model: "
                f"{self.v3_model_path}"
            )

    def load_models(self) -> None:
        """
        Load both models once.

        The lock prevents duplicate model loading if multiple
        requests arrive during application startup.
        """

        if (
            self._v1_model is not None
            and self._v3_model is not None
        ):
            return

        with self._load_lock:
            if self._v1_model is None:
                self._v1_model = (
                    tf.keras.models.load_model(
                        self.v1_model_path,
                        compile=False,
                    )
                )

            if self._v3_model is None:
                self._v3_model = (
                    tf.keras.models.load_model(
                        self.v3_model_path,
                        compile=False,
                    )
                )

    @staticmethod
    def decode_image(
        image_bytes: bytes,
    ) -> np.ndarray:
        """
        Decode uploaded image bytes into a BGR OpenCV image.
        """

        if not image_bytes:
            raise ValueError(
                "Uploaded image is empty."
            )

        encoded = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            encoded,
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise ValueError(
                "Uploaded file is not a valid image."
            )

        if image.ndim not in (2, 3):
            raise ValueError(
                f"Unsupported image dimensions: "
                f"{image.shape}"
            )

        return image

    @staticmethod
    def prepare_v1_input(
        image_bytes: bytes,
    ) -> tf.Tensor:
        """
        Reproduce the frozen V1 evaluation pipeline exactly:

        original JPEG bytes
        -> tf.io.decode_jpeg(channels=3)
        -> tf.image.resize(..., bilinear)
        -> float32
        -> batch dimension

        V1 performs 1/255 rescaling inside the model.
        """

        image = tf.io.decode_jpeg(
            image_bytes,
            channels=3,
        )

        image = tf.image.resize(
            image,
            IMAGE_SIZE,
            method="bilinear",
        )

        image = tf.cast(
            image,
            tf.float32,
        )

        return tf.expand_dims(
            image,
            axis=0,
        )

    @staticmethod
    def prepare_v3_input(
        image: np.ndarray,
    ) -> tuple[
        tf.Tensor,
        dict[str, Any],
    ]:
        """
        Reproduce the frozen V3 cached evaluation pipeline:

        decoded source image
        -> preprocess_xray()
        -> JPEG encode at quality 95
        -> tf.io.decode_jpeg(channels=3)
        -> float32
        -> batch dimension
        """

        processed, metadata = preprocess_xray(
            image,
            target_size=IMAGE_SIZE[0],
            return_metadata=True,
        )

        success, encoded = cv2.imencode(
            ".jpg",
            processed,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        if not success:
            raise RuntimeError(
                "Could not reproduce V3 cached "
                "JPEG preprocessing."
            )

        image = tf.io.decode_jpeg(
            encoded.tobytes(),
            channels=3,
        )

        image = tf.ensure_shape(
            image,
            (
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                3,
            ),
        )

        image = tf.cast(
            image,
            tf.float32,
        )

        return (
            tf.expand_dims(
                image,
                axis=0,
            ),
            metadata,
        )

    def predict_bytes(
        self,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        """
        Run the frozen production decision policy.
        """

        self.load_models()

        if (
            self._v1_model is None
            or self._v3_model is None
        ):
            raise RuntimeError(
                "Models failed to load."
            )

        image = self.decode_image(
            image_bytes
        )

        v1_input = self.prepare_v1_input(
            image_bytes
        )

        v3_input, preprocessing_metadata = (
            self.prepare_v3_input(
                image
            )
        )

        v1_probability = float(
            self._v1_model(
                v1_input,
                training=False,
            )
            .numpy()
            .reshape(-1)[0]
        )

        v3_probability = float(
            self._v3_model(
                v3_input,
                training=False,
            )
            .numpy()
            .reshape(-1)[0]
        )

        v1_positive = (
            v1_probability
            >= self.v1_threshold
        )

        v3_positive = (
            v3_probability
            >= self.v3_threshold
        )

        manual_review_warning = (
            not v1_positive
            and v3_positive
        )

        primary_label = (
            "PNEUMONIA"
            if v1_positive
            else "NORMAL"
        )

        return {
            "primary_prediction": {
                "model": "baseline_cnn_v1",
                "label": primary_label,
                "probability": v1_probability,
                "threshold": self.v1_threshold,
            },
            "secondary_signal": {
                "model": "advanced_v3",
                "role": "exploratory",
                "probability": v3_probability,
                "threshold": self.v3_threshold,
                "predicted_label": (
                    "PNEUMONIA"
                    if v3_positive
                    else "NORMAL"
                ),
                "automatic_override_allowed": False,
            },
            "decision": {
                "final_label": primary_label,
                "source": "baseline_cnn_v1",
                "manual_review_recommended": (
                    manual_review_warning
                ),
                "warning_code": (
                    "V1_NORMAL_V3_PNEUMONIA"
                    if manual_review_warning
                    else None
                ),
            },
            "preprocessing": {
                "v1": (
                    "rgb_bilinear_resize_224"
                ),
                "v3": (
                    "artifact_aware_preprocess_xray"
                ),
                "v3_metadata": (
                    preprocessing_metadata
                ),
            },
            "disclaimer": (
                "Educational decision-support prototype. "
                "Not for clinical use. Human review required."
            ),
        }


inference_service = InferenceService()
