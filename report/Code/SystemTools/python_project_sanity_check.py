"""
Project-relevant Python "Hello World" sanity check for the Jetson deployment.

Goal:
  - Confirm Python starts
  - Confirm OpenCV + NumPy import correctly
  - Confirm camera access works (read a single frame)

Optional (placeholder):
  - Load the final detection runtime (model/engine) once the deployment choice is fixed.
"""

import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")

    try:
        import cv2
    except Exception as exc:
        print(f"ERROR: failed to import OpenCV (cv2): {exc}")
        return 2

    try:
        import numpy as np  # noqa: F401
    except Exception as exc:
        print(f"ERROR: failed to import NumPy: {exc}")
        return 3

    print(f"OpenCV: {cv2.__version__}")

    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {camera_index}")
        return 4

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("ERROR: failed to read a frame from the camera")
        return 5

    height, width = frame.shape[:2]
    print(f"Camera OK: frame {width}x{height}")

    # Placeholder for model/engine load check (fill in after final deployment decision):
    #   - YOLOv5: torch.hub.load(...) or direct model loading
    #   - YOLOv8: from ultralytics import YOLO; YOLO(MODEL_PATH)
    #   - TensorRT: load .engine with TensorRT runtime
    #
    # MODEL_PATH = "TBD_MODEL_PATH"
    # print(f"Model path: {MODEL_PATH}")

    print("Python sanity check: SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

