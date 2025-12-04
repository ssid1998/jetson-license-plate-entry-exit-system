"""
Live camera test using YOLOv8n.

Opens the default MacBook camera, runs YOLOv8n inference frame by frame,
draws bounding boxes with labels and confidence, overlays FPS, and exits on 'q'.
"""

import time
from typing import Tuple

import cv2
from ultralytics import YOLO


def draw_fps(frame, fps: float) -> None:
    """Draw the FPS counter on the frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_boxes(frame, results) -> None:
    """Draw YOLO bounding boxes and labels on the frame."""
    for box in results.boxes:
        # Convert tensor values to ints for OpenCV drawing
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{results.names.get(cls_id, 'obj')} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    """Run YOLOv8n inference on the default camera feed."""
    # Load pretrained YOLOv8n weights
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from camera.")
            break

        # Run inference; results is a list with one item per image
        results = model(frame, verbose=False)[0]

        # Draw detections and FPS
        draw_boxes(frame, results)
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        draw_fps(frame, fps)

        # Show the frame
        cv2.imshow("YOLOv8n Live", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")


if __name__ == "__main__":
    main()