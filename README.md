# Car License Plate Identification with Jetson Nano

<img src="Poster/images/JetsonNanoPoster.png" alt="Jetson Nano ANPR Poster" width="750" />

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
