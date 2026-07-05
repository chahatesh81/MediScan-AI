from backend.app.ml.callbacks import build_training_callbacks


def main() -> None:
    print("=" * 60)
    print("MEDISCAN AI — CALLBACK SYSTEM TEST")
    print("=" * 60)

    callbacks, experiment_dir = build_training_callbacks(
        "baseline_cnn_test"
    )

    callback_names = [
        callback.__class__.__name__
        for callback in callbacks
    ]

    print(f"Experiment directory: {experiment_dir}")

    print("\nCallbacks:")

    for name in callback_names:
        print(f"  {name}")

    required_callbacks = {
        "ModelCheckpoint",
        "EarlyStopping",
        "ReduceLROnPlateau",
        "CSVLogger",
        "TensorBoard",
        "TerminateOnNaN",
    }

    assert required_callbacks.issubset(
        set(callback_names)
    )

    assert experiment_dir.exists()

    print("\nCALLBACK SYSTEM STATUS: READY")


if __name__ == "__main__":
    main()