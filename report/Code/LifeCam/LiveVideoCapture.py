## @file LifeCam.py
#  @brief Real-time video capture using Microsoft LifeCam on Jetson Nano.
#  @details
#  This script initializes the default camera (LifeCam Show), captures video frames in real time,
#  and displays the feed in an OpenCV window. The stream continues until the user presses the **'q'** key.
#
#  **Workflow:**
#  1. Initialize the camera capture.
#  2. Continuously read frames.
#  3. Display each frame in a named window.
#  4. Exit the loop when 'q' is pressed.
#
#  **Usage Example:**
#  @code
#  python3 LifeCam.py
#  @endcode
#
#  **Hardware Requirement:**
#  - Microsoft LifeCam Show (USB)
#
#  **Dependencies:**
#  - OpenCV (cv2)
#
import cv2

## @brief Opens the default camera and displays the live video feed.
def main():
    # Initialize video capture from the default camera (0)
    cap = cv2.VideoCapture(0)

    # Verify camera initialization
    if not cap.isOpened():
        print("Error: Unable to access LifeCam.")
        return

    print("LifeCam stream started. Press 'q' to exit.")

    # Continuous frame capture loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Frame capture failed.")
            break

        # Display the current frame in a window
        cv2.imshow("LifeCam Show Feed", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting Live Feed...")
            break

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()

## @brief Entry point for the script.
if __name__ == "__main__":
    main()