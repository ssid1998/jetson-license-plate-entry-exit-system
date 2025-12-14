# About This Project

This document provides a comprehensive technical overview of the **Car License Plate Identification System**. The project is designed to detect, recognize, and validate German license plates in real-time, optimized for edge devices like the NVIDIA Jetson Nano as well as standard desktop environments (macOS/Linux).

## 1. Project Overview

The system operates as a multi-stage pipeline:
1.  **Object Detection:** Locates the license plate within a video frame.
2.  **Image Processing:** Crops and validates the region of interest.
3.  **OCR (Optical Character Recognition):** Extracts alphanumeric text from the plate.
4.  **Validation Logic:** Verifies if the detected plate follows the German standard and/or visual characteristics (EU blue strip).

---

## 2. Dataset Details

The model was trained on a specific dataset sourced from the Roboflow Universe.

*   **Name:** `german-license-plates`
*   **Version:** v7 (Exported 2023-05-10)
*   **Source:** [Roboflow Universe - max-mustermann-gmm7j](https://universe.roboflow.com/max-mustermann-gmm7j/german-license-plates-hptbz/dataset/7)
*   **Total Images:** 1,243
*   **Classes:** Single class (`license-plates`).
*   **Format:** YOLOv8 (images + `.txt` labels).
*   **Preprocessing:** Auto-orientation of pixel data (stripping EXIF).
*   **Augmentations:** None applied during export.

**Directory Structure:**
*   `data/train`: Training set
*   `data/test`: Test set
*   `data/valid`: Validation set
*   `data/data.yaml`: YOLO configuration file defining paths and class names.

---

## 3. Machine Learning Models

### A. Object Detection (YOLOv8)
The core detection engine uses the **YOLOv8** (You Only Look Once) architecture, specifically the "Small" variant (`yolov8s.pt`), which offers a balance between speed and accuracy suitable for edge deployment.

*   **Framework:** `ultralytics`
*   **Base Model:** `yolov8s.pt` (Pre-trained on COCO dataset).
*   **Training Configuration:**
    *   **Epochs:** 10 (Short fine-tuning run).
    *   **Batch Size:** 8.
    *   **Image Size:** 640x640 pixels.
    *   **Output Directory:** `german_lpr_trained/`.

### B. Optical Character Recognition (OCR)
Text extraction is handled by **EasyOCR**, a deep-learning-based OCR library.

*   **Library:** `easyocr`
*   **Languages:** Configured for English (`['en']`) to capture standard alphanumeric characters found on plates.
*   **Input:** Cropped license plate images from the YOLO detection stage.

---

## 4. Application Logic & Algorithms

The application logic resides in the `jetson_app/` directory.

### Detection Loop (`detector.py`)
1.  Captures video feed from the camera (Index 0).
2.  Passes frames to the fine-tuned YOLOv8 model.
3.  Extracts bounding boxes for detected plates.
4.  Crops the image to the bounding box coordinates.
5.  Passes the crop to the OCR engine and the Validation module.
6.  Annotates the frame with Green (German) or Red (Non-German) bounding boxes and text.

### Country Validation (`country_check.py`)
Determines if a plate is likely German using a hybrid approach (Visual + Syntactic):

1.  **Text Cleaning:**
    *   Removes whitespace and special characters.
    *   Normalizes common OCR errors (e.g., swapping '0' for 'O' in the region code).
    *   Heuristically inserts hyphens if missing.

2.  **Regex Matching:**
    *   Pattern: `^[A-ZÄÖÜ]{1,3}[- ]?[A-Z]{1,2}[- ]?[0-9]{1,4}$`
    *   Matches standard German format: City Code (1-3 letters) + Separator + Letters (1-2) + Numbers (1-4).

3.  **Visual Verification (Blue Strip):**
    *   Analyzes the left 25% of the plate image.
    *   Uses **HSV Color Thresholding** to detect the characteristic EU blue strip.
    *   Thresholds: Blue pixels > 3% of the analyzed area.

4.  **Decision Logic:**
    *   The plate is classified as **German (DE)** if either the **Regex matches** OR the **Blue Strip is detected**.

---

## 5. Hardware & Requirements

*   **Operating System:** macOS (Darwin) or Linux (Jetson Nano/Ubuntu).
*   **Python Version:** Python 3.8+ recommended.
*   **Key Dependencies:**
    *   `ultralytics` (YOLOv8)
    *   `easyocr` & `torch` (OCR)
    *   `opencv-python` (Image processing)
    *   `numpy` (Numerical operations)
