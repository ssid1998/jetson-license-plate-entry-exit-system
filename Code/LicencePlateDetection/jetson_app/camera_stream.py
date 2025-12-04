# Camera stream logic
# TODO: Add camera stream code
from ultralytics import YOLO
import cv2

def main():
    # Load your trained model
    model = YOLO("german_lpr_trained/train2/weights/best.pt")

    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not detected")
        return

    print("🎥 Camera started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO prediction
        results = model.predict(frame, conf=0.5)

        # Draw results
        annotated = results[0].plot()

        cv2.imshow("German Plate Detection", annotated)

        # Quit on Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()