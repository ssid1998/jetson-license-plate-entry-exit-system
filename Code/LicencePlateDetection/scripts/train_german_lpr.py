from ultralytics import YOLO


def main() -> None:
    """Load the base YOLOv8n model and kick off a short fine-tuning run."""
    # Load the lightweight pretrained model as a starting point.
    model = YOLO("yolov8s.pt")

    # Launch training; adjust params here if you want longer runs or different batch sizes.
    results = model.train(
        data="data/data.yaml",
        epochs=10,
        imgsz=640,
        batch=8,
        project="german_lpr_trained",  # output folder for this experiment
        name="train",  # subfolder name inside the project directory
    )

    # Infer the save directory from the trainer if available; fall back to results.
    save_dir = None
    if hasattr(model, "trainer") and model.trainer is not None:
        save_dir = getattr(model.trainer, "save_dir", None)
    if save_dir is None and hasattr(results, "save_dir"):
        save_dir = results.save_dir  # type: ignore[attr-defined]

    print("Training run complete.")
    if save_dir:
        print(f"Results saved to: {save_dir}")
    else:
        print("Results directory could not be determined from the trainer.")


if __name__ == "__main__":
    main()