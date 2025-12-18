import os
import time

import cv2
import torch
from ultralytics import YOLO

from ocr_pipeline import PlateOcrPipeline


def main():
    print("Loading YOLO model...")
    modelPath = os.path.join(
        os.path.dirname(__file__),
        "..",
        "german_lpr_trained",
        "train2",
        "weights",
        "best.pt",
    )
    modelPath = os.path.abspath(modelPath)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    yoloModel = YOLO(modelPath)
    try:
        yoloModel.to(device)
    except Exception:
        print("Warning: could not move model to requested device, using default.")

    print("Initializing plate OCR pipeline...")
    ocrPipeline = PlateOcrPipeline(
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        confThreshold=0.6,
        euMaskRatio=0.15,
        debug=False,
        useGpu=torch.cuda.is_available(),
    )
    captureDir = os.path.join(os.path.dirname(__file__), "captures")
    os.makedirs(captureDir, exist_ok=True)
    captureConfidence = 0.5
    minCaptureIntervalMs = 1500
    lastCaptureMs = 0
    detectConf = 0.2  # lower if you see nothing; raise if too many false positives
    detectIou = 0.5

    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Camera not detected")
        return

    print("Real-time German Plate Detection + OCR")
    print("Press 'q' to quit.")

    while True:
        retVal, frameImg = cap.read()
        if not retVal:
            break

        results = yoloModel(
            frameImg,
            imgsz=640,
            device=device,
            verbose=False,
            conf=detectConf,
            iou=detectIou,
        )[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            frameHeight, frameWidth = frameImg.shape[:2]
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            x2, y2 = max(int(x2), 0), max(int(y2), 0)
            x2, y2 = min(x2, frameWidth), min(y2, frameHeight)

            plateCrop = frameImg[y1:y2, x1:x2]

            plateText, plateConf, debugInfo = ocrPipeline.readPlate(plateCrop)
            ocrPipeline.updateTemporalVote(plateText, plateConf)
            smoothText, smoothConf = ocrPipeline.getTemporalConsensus()

            displayText = smoothText or plateText
            displayConf = smoothConf if smoothText else plateConf

            label = displayText if displayText else "plate"
            if displayConf:
                label = f"{label} ({displayConf:.2f})"

            cv2.rectangle(frameImg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frameImg,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if ocrPipeline.debug:
                print(f"Debug info: {debugInfo}")

            boxConf = float(box.conf[0]) if hasattr(box, "conf") and len(box.conf) > 0 else 0.0
            nowMs = int(time.time() * 1000)
            if boxConf >= captureConfidence and (nowMs - lastCaptureMs) >= minCaptureIntervalMs:
                ts = int(time.time() * 1000)
                framePath = os.path.join(captureDir, f"frame_{ts}.jpg")
                platePath = os.path.join(captureDir, f"plate_{ts}.jpg")
                cv2.imwrite(framePath, frameImg)
                if plateCrop.size > 0:
                    cv2.imwrite(platePath, plateCrop)
                lastCaptureMs = nowMs

        if len(results.boxes) == 0:
            cv2.putText(
                frameImg,
                "No plate detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow("German Plate Detection + OCR", frameImg)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
