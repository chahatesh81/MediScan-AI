import numpy as np
import tensorflow as tf


def find_last_conv_layer(
    model: tf.keras.Model,
) -> str:
    """Find the last Conv2D layer in the model."""

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError("No Conv2D layer found in model.")


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    image_batch: tf.Tensor,
    last_conv_layer_name: str | None = None,
    return_mode: bool = False,
    output_layer_name: str = "pneumonia_probability",
    target_class_index: int = 0,
) -> np.ndarray | tuple[np.ndarray, str]:
    """
    Generate a normalized attribution heatmap.

    Standard positive-evidence Grad-CAM is used first.
    If the positive map is entirely zero, the function
    falls back to absolute attribution magnitude.
    """

    if image_batch.shape[0] != 1:
        raise ValueError(
            "Grad-CAM currently expects a batch size of 1."
        )

    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    last_conv_layer = model.get_layer(
        last_conv_layer_name
    )

    output_layer = model.get_layer(
        output_layer_name
    )

    output_units = int(output_layer.units)

    if not 0 <= target_class_index < output_units:
        raise ValueError(
            "Grad-CAM target class index is out of range: "
            f"{target_class_index}; expected 0 to "
            f"{output_units - 1}."
        )

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            output_layer.input,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, classifier_features = grad_model(
            image_batch,
            training=False,
        )

        classifier_features_fp32 = tf.cast(
            classifier_features,
            tf.float32,
        )

        kernel_fp32 = tf.cast(
            output_layer.kernel,
            tf.float32,
        )

        logits = tf.matmul(
            classifier_features_fp32,
            kernel_fp32,
        )

        if output_layer.bias is not None:
            logits = tf.nn.bias_add(
                logits,
                tf.cast(
                    output_layer.bias,
                    tf.float32,
                ),
            )

        target_score = logits[
            :,
            target_class_index,
        ]

    gradients = tape.gradient(
        target_score,
        conv_outputs,
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients could not be computed."
        )

    conv_outputs_fp32 = tf.cast(
        conv_outputs,
        tf.float32,
    )

    gradients_fp32 = tf.cast(
        gradients,
        tf.float32,
    )

    gradients_are_finite = tf.reduce_all(
        tf.math.is_finite(gradients_fp32)
    )

    if not bool(gradients_are_finite.numpy()):
        raise RuntimeError(
            "Grad-CAM gradients contain NaN or infinity."
        )

    pooled_gradients = tf.reduce_mean(
        gradients_fp32,
        axis=(0, 1, 2),
    )

    feature_maps = conv_outputs_fp32[0]

    raw_heatmap = tf.reduce_sum(
        feature_maps
        * pooled_gradients[
            tf.newaxis,
            tf.newaxis,
            :
        ],
        axis=-1,
    )

    heatmap_is_finite = tf.reduce_all(
        tf.math.is_finite(raw_heatmap)
    )

    if not bool(heatmap_is_finite.numpy()):
        raise RuntimeError(
            "Grad-CAM heatmap contains NaN or infinity."
        )

    positive_heatmap = tf.nn.relu(
        raw_heatmap
    )

    positive_maximum = tf.reduce_max(
        positive_heatmap
    )

    if float(positive_maximum.numpy()) > 0.0:
        heatmap = positive_heatmap
        maximum = positive_maximum
        explanation_mode = "positive_gradcam"
    else:
        heatmap = tf.abs(raw_heatmap)
        maximum = tf.reduce_max(heatmap)
        explanation_mode = "absolute_attribution"

    maximum_value = float(maximum.numpy())

    if maximum_value <= 0.0:
        raise RuntimeError(
            "Grad-CAM attribution is numerically zero."
        )

    heatmap = heatmap / maximum

    heatmap_array = (
        heatmap
        .numpy()
        .astype(np.float32)
    )

    if not np.isfinite(heatmap_array).all():
        raise RuntimeError(
            "Final Grad-CAM heatmap contains invalid values."
        )

    if return_mode:
        return heatmap_array, explanation_mode

    return heatmap_array