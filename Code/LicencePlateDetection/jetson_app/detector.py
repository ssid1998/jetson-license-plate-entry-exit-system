# detector.py

from ultralytics import YOLO
import cv2
import easyocr

from country_check import is_german_plate


def main():
    print("Loading YOLO model...")
    model = YOLO("german_lpr_trained/train2/weights/best.pt")

    print("Loading OCR engine (EasyOCR)...")
    reader = easyocr.Reader(['en'], gpu=False)

    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Camera not detected")
        return

    print("🎥 Real-time German Plate Detection + Country Validation")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop license plate region
            plate_crop = frame[y1:y2, x1:x2]

            # OCR using EasyOCR
            ocr_result = reader.readtext(plate_crop)
            raw_text = ocr_result[0][1] if len(ocr_result) > 0 else ""

            # Country validation (calls country_check.py)
            is_de, display_text = is_german_plate(plate_crop, raw_text)

            if is_de:
                label = f"DE ✓ {display_text}"
                color = (0, 255, 0)
            else:
                label = f"NOT DE ✗ {display_text}"
                color = (0, 0, 255)

            # Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        cv2.imshow("German Plate Detection + Country Validation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()