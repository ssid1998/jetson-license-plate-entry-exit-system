# Car License Plate Identification with Jetson Nano

## 📖 Project Overview
This project implements *Automatic Number Plate Recognition (ANPR)* on the *NVIDIA Jetson Nano*.  
It leverages *deep learning–based object detection* methods (e.g., YOLO, CNNs) and *OCR techniques* to detect and recognize license plates in real time.  

The work is part of a university project and focuses on the following goals:
- Deploy a license plate detection pipeline optimized for edge AI devices.
- Compare different deep learning models (YOLO, SSD, Faster R-CNN) on embedded hardware.
- Explore OCR-based text extraction for license plate numbers.
- Demonstrate real-time inference on Jetson Nano.

---

## 📂 Folder Structure
A simplified view of the repository:
├── Assembly/                # LaTeX report build files
├── Code/                    # Source code
│   ├── Arduino/             # Arduino scripts
│   ├── MicroPython/         # MicroPython scripts
│   └── JetsonNano/          # Jetson Nano-specific code
│       ├── Code/            # Core Python detection scripts
│       ├── Images/          # Sample input/output images
│       ├── System/          # Configuration & system setup
│       └── Contents/        # Supporting files
├── Manual/                  # Project manual (LaTeX chapters)
├── MLLib/                   # Literature, manuals, and references
├── Poster/                  # TikZ poster files
├── Presentations/            # Presentation slides
│   └── Literatures/          # Literature review & references
│       ├── LiteratureReview.pdf
│       └── References/       # Collection of research PDFs
├── ProjectManagement/       # Checklists, evaluation docs
├── report/                  # Final written report
└── README.md                # This file
Presentations/Literatures/References/
Key sources cover:
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
