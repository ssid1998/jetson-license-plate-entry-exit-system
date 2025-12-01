#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
! \file OpenCVExample.py
! \brief Minimal OpenCV test script for Jetson Nano licence plate project.
!
! This script demonstrates how to:
!   - open a camera stream using OpenCV,
!   - perform basic image pre-processing (grayscale conversion),
!   - draw a static region of interest (ROI) rectangle,
!   - display the processed frames in a window,
!   - and cleanly release all resources.
!
! It is intended as a sanity check that the camera,
! OpenCV installation, and basic image-processing pipeline
! work correctly on the Jetson Nano B01.
"""

import argparse
from typing import Tuple

import cv2


def parse_arguments() -> argparse.Namespace:
    """
    ! \brief Parse command-line arguments.
    !
    ! This allows the user to adjust basic runtime parameters
    ! without changing the source code.
    !
    ! \return argparse.Namespace Structure containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "OpenCV camera sanity-check for the licence plate detection project. "
            "Opens a camera stream, converts frames to grayscale, draws a static "
            "ROI rectangle and displays the result."
        )
    )

    parser.add_argument(
        "-c",
        "--camera-index",
        type=int,
        default=0,
        help="Index of the camera device (default: 0). "
             "For USB cameras use 0/1; for CSI cameras a different backend may be required.",
    )

    parser.add_argument(
        "-W",
        "--width",
        type=int,
        default=1280,
        help="Desired capture width in pixels (default: 1280).",
    )

    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=720,
        help="Desired capture height in pixels (default: 720).",
    )

    parser.add_argument(
        "--no-gray",
        action="store_true",
        help="Disable grayscale conversion and show colour frames instead.",
    )

    parser.add_argument(
        "--window-name",
        type=str,
        default="OpenCV Camera Test",
        help="Name of the display window (default: 'OpenCV Camera Test').",
    )

    return parser.parse_args()


def initialise_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    """
    ! \brief Initialise an OpenCV VideoCapture object.
    !
    ! This function tries to open the specified camera index and
    ! sets the desired frame width and height. It raises a RuntimeError
    ! if the camera cannot be opened.
    !
    ! \param camera_index Index of the camera device to open.
    ! \param width Desired frame width in pixels.
    ! \param height Desired frame height in pixels.
    ! \return cv2.VideoCapture Opened and configured VideoCapture object.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"ERROR: Could not open camera with index {camera_index}."
        )

    # Request specific resolution (may not be honoured by all cameras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    return cap


def compute_roi_rectangle(
    frame_shape: Tuple[int, int, int],
    roi_scale: float = 0.5,
) -> Tuple[int, int, int, int]:
    """
    ! \brief Compute a centred ROI rectangle for licence plate region.
    !
    ! The ROI is defined as a rectangle in the lower-central part
    ! of the frame, where licence plates are likely to appear in a
    ! typical side/front view scenario.
    !
    ! \param frame_shape Shape of the frame as (height, width, channels).
    ! \param roi_scale Fraction of the frame width used for the ROI width
    !        (0 < roi_scale <= 1). The ROI height is chosen proportionally.
    ! \return (x1, y1, x2, y2) Coordinates of the top-left and bottom-right
    !         corners of the ROI rectangle.
    """
    height, width, _ = frame_shape

    roi_width = int(width * roi_scale)
    roi_height = int(height * 0.25)  # bottom quarter-ish

    x1 = int((width - roi_width) / 2)
    x2 = x1 + roi_width

    # Place ROI in the lower third of the image
    y2 = int(height * 0.9)
    y1 = y2 - roi_height

    return x1, y1, x2, y2


def process_frame(
    frame,
    use_gray: bool = True,
    draw_roi: bool = True,
) -> Tuple:
    """
    ! \brief Apply basic processing steps to a single frame.
    !
    ! The processing includes:
    !   - optional conversion to grayscale,
    !   - drawing a static ROI rectangle (if enabled).
    !
    ! \param frame Input frame (BGR) from the camera.
    ! \param use_gray If True, the output is converted to grayscale.
    ! \param draw_roi If True, a static ROI rectangle is drawn on the frame.
    ! \return (processed_frame, roi_coords) Tuple containing the processed
    !         frame and the ROI coordinates as (x1, y1, x2, y2). If no ROI
    !         is drawn, roi_coords is None.
    """
    roi_coords = None

    if use_gray:
        # Convert from BGR (OpenCV default) to grayscale
        processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # For display, convert back to BGR so that the window type is consistent
        processed_display = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    else:
        processed = frame.copy()
        processed_display = processed

    if draw_roi:
        roi_coords = compute_roi_rectangle(processed_display.shape, roi_scale=0.5)
        x1, y1, x2, y2 = roi_coords

        # Draw a rectangle representing the expected licence plate region
        cv2.rectangle(
            processed_display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),  # green rectangle
            2,
        )

        # Optional label
        cv2.putText(
            processed_display,
            "ROI: Licence Plate Region",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return processed_display, roi_coords


def main() -> None:
    """
    ! \brief Main entry point of the OpenCV example.
    !
    ! This function:
    !   - prints the OpenCV version,
    !   - initialises the camera,
    !   - enters a loop to capture and process frames,
    !   - displays the processed frames in a window,
    !   - and exits when the user presses the 'q' key.
    """
    args = parse_arguments()

    print("Starting OpenCV camera test...")
    print("OpenCV version:", cv2.__version__)
    print(f"Using camera index: {args.camera_index}")
    print(f"Requested resolution: {args.width} x {args.height}")
    print("Grayscale mode:", not args.no_gray)
    print("Press 'q' in the window to quit.\n")

    try:
        cap = initialise_camera(
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
        )
    except RuntimeError as err:
        print(err)
        return

    window_name = args.window_name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("WARNING: Failed to read frame from camera.")
                break

            processed_frame, roi_coords = process_frame(
                frame,
                use_gray=not args.no_gray,
                draw_roi=True,
            )

            # Display the result
            cv2.imshow(window_name, processed_frame)

            # Wait for 1 ms and check for 'q' key to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("INFO: 'q' pressed, exiting.")
                break

    finally:
        # Always release resources, even if an exception occurs
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and all windows closed.")


if __name__ == "__main__":
    main()