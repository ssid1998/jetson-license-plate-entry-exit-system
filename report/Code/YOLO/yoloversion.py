"""
YOLO and PyTorch Version Check
Author: Shubhankar Dumka
"""

import torch
import ultralytics
from ultralytics import YOLO


def main():
    print("Ultralytics version:", ultralytics.__version__)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
    