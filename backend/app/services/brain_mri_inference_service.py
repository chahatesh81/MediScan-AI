from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from backend.app.ml.brain_mri.dataset_loader import (
    ImagePreprocessingConfig,
    decode_image_payload,
)
from backend.app.ml.brain_mri.model_training import (
    DEFAULT_FINAL_MODEL_PATH,
    class_labels_by_index,
)


class BrainMRIInferenceError(RuntimeError):
    """Base error for Brain MRI inference failures."""


class BrainMRIModelArtifactNotFoundError(
    BrainMRIInferenceError
):
    """Raised when the production model artifact is missing."""


class BrainMRIModelContractError(
    BrainMRIInferenceError
):
    """Raised when the saved model violates its contract."""


class BrainMRIInferenceService:
    """Production inference service for Brain MRI classification."""

    def __init__(
        self,
        model_path: Path = DEFAULT_FINAL_MODEL_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self._load_lock = threading.Lock()
        self._model: tf.keras.Model | None = None

        self._class_labels = (
            class_labels_by_index()
        )

        self._preprocessing_config = (
            ImagePreprocessingConfig(
                height=224,
                width=224,
                channels=3,
            )
        )

    @property
    def class_labels(self) -> tuple[str, ...]:
        return self._class_labels

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            if not self.model_path.is_file():
                raise (
                    BrainMRIModelArtifactNotFoundError(
                        "Brain MRI model artifact "
                        f"does not exist: {self.model_path}"
                    )
                )

            model = tf.keras.models.load_model(
                self.model_path,
                compile=False,
            )

            self._validate_model_contract(
                model
            )

            self._model = model

    def _validate_model_contract(
        self,
        model: tf.keras.Model,
    ) -> None:
        expected_input_shape = (
            None,
            224,
            224,
            3,
        )

        expected_output_shape = (
            None,
            len(self._class_labels),
        )

        if tuple(model.input_shape) != (
            expected_input_shape
        ):
            raise BrainMRIModelContractError(
                "Unexpected Brain MRI model "
                f"input shape: {model.input_shape}"
            )

        if tuple(model.output_shape) != (
            expected_output_shape
        ):
            raise BrainMRIModelContractError(
                "Unexpected Brain MRI model "
                f"output shape: {model.output_shape}"
            )

    def prepare_input(
        self,
        image_bytes: bytes,
    ) -> np.ndarray:
        image = decode_image_payload(
            image_bytes,
            config=self._preprocessing_config,
        )

        batch = np.expand_dims(
            image,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        expected_shape = (
            1,
            224,
            224,
            3,
        )

        if batch.shape != expected_shape:
            raise ValueError(
                "Unexpected Brain MRI input shape: "
                f"{batch.shape}"
            )

        return batch

    def predict_bytes(
        self,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        self.load_model()

        if self._model is None:
            raise BrainMRIInferenceError(
                "Brain MRI model failed to load."
            )

        model_input = self.prepare_input(
            image_bytes
        )

        raw_output = self._model(
            model_input,
            training=False,
        )

        probabilities = np.asarray(
            raw_output.numpy(),
            dtype=np.float64,
        )

        expected_shape = (
            1,
            len(self._class_labels),
        )

        if probabilities.shape != expected_shape:
            raise BrainMRIModelContractError(
                "Unexpected Brain MRI prediction "
                f"shape: {probabilities.shape}"
            )

        probability_vector = probabilities[0]

        if not np.all(
            np.isfinite(probability_vector)
        ):
            raise BrainMRIModelContractError(
                "Brain MRI prediction contains "
                "non-finite probabilities."
            )

        probability_sum = float(
            np.sum(probability_vector)
        )

        if not np.isclose(
            probability_sum,
            1.0,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise BrainMRIModelContractError(
                "Brain MRI probabilities do not "
                "sum to 1."
            )

        predicted_index = int(
            np.argmax(probability_vector)
        )

        predicted_label = (
            self._class_labels[
                predicted_index
            ]
        )

        scores = tuple(
            {
                "label": label,
                "probability": float(
                    probability_vector[index]
                ),
            }
            for index, label in enumerate(
                self._class_labels
            )
        )

        return {
            "predicted_label": predicted_label,
            "scores": scores,
        }


brain_mri_inference_service = (
    BrainMRIInferenceService()
)
