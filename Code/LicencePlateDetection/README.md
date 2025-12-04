# Jetson License Plate Recognition

This project provides a pipeline for license plate detection and recognition using YOLO and OCR, designed for Jetson Nano and MacBook environments.

## Structure
- `data/`: Datasets and config
- `models/`: YOLO models
- `notebooks/`: Analysis and preview
- `scripts/`: Training, testing, and inference scripts
- `jetson_app/`: Jetson Nano application

## Setup
1. Create and activate a Python virtual environment in `.venv`
2. Install dependencies: `pip install -r requirements.txt`
3. Add your models and data as needed

## Usage
- Run scripts for training, testing, and inference
- Use `jetson_app/main.py` for the Jetson Nano application
