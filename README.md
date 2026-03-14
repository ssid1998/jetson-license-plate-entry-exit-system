# Automated Vehicle Entry–Exit Analytics System  
*(Edge AI License Plate Recognition using NVIDIA Jetson Nano)*

<img src="Poster/images/JetsonNanoPoster.png" alt="Jetson Nano ANPR Poster" width="750" />

## 📖 Project Overview

This project implements an **Automated Vehicle Entry–Exit Analytics System** using **edge AI on the NVIDIA Jetson Nano**.

The system performs **Automatic Number Plate Recognition (ANPR)** to detect vehicles entering and exiting a facility by identifying license plates in real time.

Using **deep learning–based object detection and OCR pipelines**, the project demonstrates how **embedded edge devices can perform real-time vehicle monitoring and access analytics** without relying on cloud processing.

The project focuses on:

- Detecting vehicle license plates using **deep learning object detection models**
- Extracting license plate text using **Optical Character Recognition (OCR)**
- Running the full inference pipeline **on the NVIDIA Jetson Nano edge device**
- Evaluating **performance trade-offs of different detection architectures** on embedded hardware

---

## ⚙ System Capabilities

The system enables:

- Automated vehicle detection at **entry and exit points**
- **License plate recognition** using deep learning + OCR
- **Edge inference** using Jetson Nano (low latency)
- **Vehicle entry–exit monitoring and analytics**

Example applications include:

- Smart parking systems
- Automated gate access control
- Traffic monitoring
- Campus or facility vehicle logging

---

## 🧠 AI / Computer Vision Pipeline

The system pipeline follows these stages:

1. **Image Capture**
   - Camera input connected to the Jetson Nano

2. **License Plate Detection**
   - Deep learning models used for detection:
     - YOLO
     - SSD
     - Faster R-CNN

3. **Plate Cropping**
   - Detected plate region is isolated

4. **Text Recognition**
   - OCR techniques extract license plate numbers

5. **Vehicle Entry–Exit Logging**
   - Recognized plate numbers can be used for tracking and analytics

---

## 🛠 Technology Stack

- **Hardware**
  - NVIDIA Jetson Nano
  - Camera module

- **Software**
  - Python
  - OpenCV
  - PyTorch / TensorFlow
  - YOLO object detection
  - OCR pipeline

- **Deployment**
  - Edge AI inference on embedded hardware

---

## 📂 Folder Structure
A simplified view of the repository:

```text
.
├── Assembly/                     # LaTeX report build files
├── Code/                         # Source code
│   └── LicencePlateDetection/    # ANPR / LPR experiments & apps
├── JetsonNano/                   # Jetson Nano-specific code & setup
│   ├── Code/                     # Scripts (Arduino/Jetson/etc.)
│   ├── Images/                   # Sample images & figures
│   ├── System/                   # Configuration & system setup
│   └── Contents/                 # Supporting files
├── Manual/                       # Project manual (LaTeX chapters)
├── MLbib/                        # Literature, manuals, and references
├── Poster/                       # TikZ poster files
├── Presentations/                # Presentation slides
│   └── Literatures/              # Literature review & references
│       ├── LiteratureReview.pdf
│       └── References/           # Collection of research PDFs
├── ProjectManagement/            # Checklists, evaluation docs
├── report/                       # Final written report
└── README.md                     # This file
```



## 📚 Literature
The literature collection under `Presentations/Literatures/References/` covers:
- Deep learning approaches for license plate detection
- YOLO, SSD, and Faster R-CNN comparisons
- OCR-based text extraction methods
- Jetson Nano and edge AI deployment studies

---

## 👥 Contributors
- Gautam Ramesh (Matriculation Number: 7026787)
- Shubhankar Dumka (Matriculation Number: 7027187)
- Siddhanth Sharma (Matriculation Number: 7027189)

---

## 📌 Notes
- This project is developed for academic purposes under the Master’s program in *Business Intelligence and Data Analytics*.
- Future work will include benchmarking model accuracy and speed, and improving OCR reliability on real-world data.
