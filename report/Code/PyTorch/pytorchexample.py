#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
! \file PyTorchLPExample.py
! \brief Minimal PyTorch test script for Jetson Nano licence plate detection.
!
! This script demonstrates how to:
!   - open a camera stream using OpenCV,
!   - convert frames to PyTorch tensors,
!   - run a pre-trained licence plate detection model,
!   - draw detected bounding boxes on frames,
!   - and display the annotated frames in a window.
!
! It serves as a basic sanity check for PyTorch inference on the Jetson Nano.
"""

import argparse
from typing import Tuple, List
import cv2
import torch
from torchvision import transforms

# Dummy placeholder model for illustration
class DummyLPDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # Simulate detection: returns one dummy box per frame
        # Format: [x1, y1, x2, y2, confidence]
        return torch.tensor([[100, 200, 400, 250, 0.95]])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch licence plate detection demo for Jetson Nano.")
    parser.add_argument("-c", "--camera-index", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("-W", "--width", type=int, default=1280, help="Capture width (default 1280).")
    parser.add_argument("-H", "--height", type=int, default=720, help="Capture height (default 720).")
    parser.add_argument("--window-name", type=str, default="PyTorch LP Demo", help="Window title.")
    return parser.parse_args()

def initialise_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"ERROR: Could not open camera {camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return cap

def preprocess_frame(frame) -> torch.Tensor:
    # Convert BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to tensor and normalize [0,1]
    tensor = transforms.ToTensor()(img).unsqueeze(0)  # shape: [1,C,H,W]
    return tensor

def annotate_frame(frame, detections: torch.Tensor) -> None:
    for box in detections:
        x1, y1, x2, y2, conf = box.int()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"LP {conf:.2f}", (x1, max(y1-10,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

def main() -> None:
    args = parse_arguments()
    print("Starting PyTorch licence plate detection demo...")
    print("CUDA available:", torch.cuda.is_available())

    try:
        cap = initialise_camera(args.camera_index, args.width, args.height)
    except RuntimeError as err:
        print(err)
        return

    window_name = args.window_name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Load dummy detection model (replace with real model in practice)
    model = DummyLPDetector()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("WARNING: Failed to read frame.")
                break

            # Preprocess frame for PyTorch
            input_tensor = preprocess_frame(frame)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

            # Run model inference
            with torch.no_grad():
                detections = model(input_tensor)  # shape: [num_boxes,5]

            # Annotate frame
            annotate_frame(frame, detections.cpu())

            # Display
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("INFO: 'q' pressed, exiting.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and all windows closed.")

if __name__ == "__main__":
    main()
