from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from backend.app.ml.gradcam import (
    find_last_conv_layer,
    generate_gradcam_heatmap,
)


def build_gradcam_model(
    *,
    output_kernel: float = 1.0,
    output_bias: float = 0.0,
) -> tf.keras.Model:
    inputs = tf.keras.Input(
        shape=(8, 8, 1),
        name="image",
    )

    x = tf.keras.layers.Conv2D(
        filters=2,
        kernel_size=3,
        padding="same",
        activation="relu",
        use_bias=False,
        kernel_initializer="ones",
        name="conv_first",
    )(inputs)

    x = tf.keras.layers.Conv2D(
        filters=2,
        kernel_size=3,
        padding="same",
        activation="relu",
        use_bias=False,
        kernel_initializer="ones",
        name="conv_last",
    )(x)

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling",
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=tf.keras.initializers.Constant(
            output_kernel
        ),
        bias_initializer=tf.keras.initializers.Constant(
            output_bias
        ),
        name="pneumonia_probability",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="gradcam_test_model",
    )


def test_find_last_conv_layer_returns_final_conv2d() -> None:
    model = build_gradcam_model()

    assert find_last_conv_layer(model) == "conv_last"


def test_find_last_conv_layer_rejects_model_without_conv2d() -> None:
    inputs = tf.keras.Input(
        shape=(4,),
        name="features",
    )
    outputs = tf.keras.layers.Dense(
        1,
        name="pneumonia_probability",
    )(inputs)
    model = tf.keras.Model(inputs, outputs)

    with pytest.raises(
        ValueError,
        match="No Conv2D layer found",
    ):
        find_last_conv_layer(model)


@pytest.mark.parametrize(
    "batch_size",
    [0, 2],
)
def test_generate_gradcam_rejects_non_singleton_batch(
    batch_size: int,
) -> None:
    model = build_gradcam_model()
    image_batch = tf.zeros(
        (batch_size, 8, 8, 1),
        dtype=tf.float32,
    )

    with pytest.raises(
        ValueError,
        match="batch size of 1",
    ):
        generate_gradcam_heatmap(
            model=model,
            image_batch=image_batch,
        )


def test_generate_gradcam_returns_normalized_positive_heatmap() -> None:
    model = build_gradcam_model(
        output_kernel=1.0,
    )
    image_batch = tf.ones(
        (1, 8, 8, 1),
        dtype=tf.float32,
    )

    heatmap, mode = generate_gradcam_heatmap(
        model=model,
        image_batch=image_batch,
        return_mode=True,
    )

    assert mode == "positive_gradcam"
    assert heatmap.ndim == 2
    assert heatmap.shape == (8, 8)
    assert heatmap.dtype == np.float32
    assert np.isfinite(heatmap).all()
    assert float(np.min(heatmap)) >= 0.0
    assert float(np.max(heatmap)) == pytest.approx(1.0)


def test_generate_gradcam_uses_absolute_attribution_fallback() -> None:
    model = build_gradcam_model(
        output_kernel=-1.0,
    )
    image_batch = tf.ones(
        (1, 8, 8, 1),
        dtype=tf.float32,
    )

    heatmap, mode = generate_gradcam_heatmap(
        model=model,
        image_batch=image_batch,
        return_mode=True,
    )

    assert mode == "absolute_attribution"
    assert heatmap.ndim == 2
    assert heatmap.shape == (8, 8)
    assert heatmap.dtype == np.float32
    assert np.isfinite(heatmap).all()
    assert float(np.min(heatmap)) >= 0.0
    assert float(np.max(heatmap)) == pytest.approx(1.0)


def test_generate_gradcam_returns_array_without_mode() -> None:
    model = build_gradcam_model()
    image_batch = tf.ones(
        (1, 8, 8, 1),
        dtype=tf.float32,
    )

    result = generate_gradcam_heatmap(
        model=model,
        image_batch=image_batch,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (8, 8)


def test_generate_gradcam_rejects_zero_attribution() -> None:
    model = build_gradcam_model(
        output_kernel=0.0,
    )
    image_batch = tf.ones(
        (1, 8, 8, 1),
        dtype=tf.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="attribution is numerically zero",
    ):
        generate_gradcam_heatmap(
            model=model,
            image_batch=image_batch,
        )


def test_generate_gradcam_rejects_zero_attribution() -> None:
    model = build_gradcam_model(
        output_kernel=0.0,
    )
    image_batch = tf.ones(
        (1, 8, 8, 1),
        dtype=tf.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="attribution is numerically zero",
    ):
        generate_gradcam_heatmap(
            model=model,
            image_batch=image_batch,
        )
