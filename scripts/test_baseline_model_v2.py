import tensorflow as tf

from backend.app.core.config import RANDOM_SEED
from backend.app.core.reproducibility import set_global_seed
from backend.app.ml.baseline_cnn import build_baseline_cnn
from backend.app.ml.baseline_cnn_v2 import build_baseline_cnn_v2


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "BASELINE V2 MODEL TEST"
    )
    print("=" * 70)

    set_global_seed(RANDOM_SEED)

    model_v1 = build_baseline_cnn()
    model_v2 = build_baseline_cnn_v2()

    v1_parameters = model_v1.count_params()
    v2_parameters = model_v2.count_params()

    test_batch = tf.zeros(
        shape=(2, 224, 224, 3),
        dtype=tf.float32,
    )

    predictions = model_v2(
        test_batch,
        training=False,
    )

    print()
    print(
        f"V1 parameters:       "
        f"{v1_parameters:,}"
    )
    print(
        f"V2 parameters:       "
        f"{v2_parameters:,}"
    )
    print(
        f"Parameter match:      "
        f"{v1_parameters == v2_parameters}"
    )
    print(
        f"V2 model name:        "
        f"{model_v2.name}"
    )
    print(
        f"Input shape:          "
        f"{model_v2.input_shape}"
    )
    print(
        f"Output shape:         "
        f"{model_v2.output_shape}"
    )
    print(
        f"Prediction shape:     "
        f"{predictions.shape}"
    )
    print(
        f"Prediction dtype:     "
        f"{predictions.dtype}"
    )

    if v1_parameters != v2_parameters:
        raise RuntimeError(
            "V1 and V2 parameter counts differ. "
            "The architecture comparison is not controlled."
        )

    if model_v2.input_shape != (
        None,
        224,
        224,
        3,
    ):
        raise RuntimeError(
            "Unexpected V2 input shape."
        )

    if model_v2.output_shape != (
        None,
        1,
    ):
        raise RuntimeError(
            "Unexpected V2 output shape."
        )

    if predictions.dtype != tf.float32:
        raise RuntimeError(
            "V2 output must remain float32."
        )

    print()
    print("=" * 70)
    print(
        "BASELINE V2 MODEL STATUS: READY"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()