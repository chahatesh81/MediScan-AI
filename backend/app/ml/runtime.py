import tensorflow as tf


def configure_training_runtime() -> dict[str, object]:
    gpus = tf.config.list_physical_devices("GPU")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )
        except RuntimeError:
            # TensorFlow may already have initialized the GPU.
            pass

    if gpus:
        tf.keras.mixed_precision.set_global_policy(
            "mixed_float16"
        )
    else:
        tf.keras.mixed_precision.set_global_policy(
            "float32"
        )

    policy = (
        tf.keras.mixed_precision.global_policy()
    )

    return {
        "gpu_count": len(gpus),
        "gpus": gpus,
        "policy": policy.name,
        "compute_dtype": policy.compute_dtype,
        "variable_dtype": policy.variable_dtype,
    }