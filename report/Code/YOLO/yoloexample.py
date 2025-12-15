"""
YOLO License Plate Detection Example for Jetson Nano
Author: Shubhankar Dumka
Description:
    Minimal YOLO inference loop using Ultralytics YOLO.
    Captures live camera frames, performs detection,
    draws bounding boxes and confidence scores.
"""

import cv2
import torch
from ultralytics import YOLO


def main():
    # Load YOLO model (use a custom trained license plate model if available)
    # Example: yolov8n.pt for demo, replace with your trained .pt file
    model = YOLO("yolov8n.pt")

    # Open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    print("YOLO model loaded")
    print("CUDA available:", torch.cuda.is_available())

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run YOLO inference
        results = model(frame, conf=0.4, iou=0.5, verbose=False)

        # Draw detections
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                label = f"Plate {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("YOLO License Plate Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
