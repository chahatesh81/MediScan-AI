import numpy as np
import tensorflow as tf

from backend.app.core.config import PROJECT_ROOT
from backend.app.ml.gradcam import (
    find_last_conv_layer,
    generate_gradcam_heatmap,
)
from backend.app.ml.runtime import (
    configure_training_runtime,
)
from backend.app.services.data_pipeline import (
    build_dataset,
)


MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "mediscan_final.keras"
)


def main() -> None:
    print("=" * 70)
    print("MEDISCAN AI — GRAD-CAM ENGINE TEST")
    print("=" * 70)

    configure_training_runtime()

    model = tf.keras.models.load_model(
        MODEL_FILE,
        compile=False,
    )

    dataset = build_dataset(
        split="test",
        batch_size=1,
        shuffle=False,
    )

    images, labels = next(
        iter(dataset)
    )

    last_conv_layer = (
        find_last_conv_layer(model)
    )

    probability = float(
        model(
            images,
            training=False,
        )[0, 0].numpy()
    )

    heatmap = generate_gradcam_heatmap(
        model=model,
        image_batch=images,
        last_conv_layer_name=last_conv_layer,
    )

    print(f"Model:            {model.name}")
    print(f"Last conv layer:  {last_conv_layer}")
    print(f"True label:       {int(labels[0].numpy())}")
    print(f"Probability:      {probability:.6f}")
    print(f"Heatmap shape:    {heatmap.shape}")
    print(f"Heatmap dtype:    {heatmap.dtype}")
    print(f"Heatmap minimum:  {heatmap.min():.6f}")
    print(f"Heatmap maximum:  {heatmap.max():.6f}")

    assert heatmap.ndim == 2
    assert heatmap.shape[0] > 0
    assert heatmap.shape[1] > 0
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6
    assert heatmap.max() > 0.0
    assert np.count_nonzero(heatmap) > 0

    print(
        "\nGRAD-CAM ENGINE STATUS: READY"
    )


if __name__ == "__main__":
    main()