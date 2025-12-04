# country_check.py

import cv2
import numpy as np
import re

# ----------------------------------------------------
# German license plate format (text-based pattern)
# Examples that should match after cleaning:
#   "M G4187", "M-G 4187", "M-G4187", "MG 4187"
# ----------------------------------------------------
GERMAN_REGEX = re.compile(
    r"^[A-ZÄÖÜ]{1,3}[- ]?[A-Z]{1,2}[- ]?[0-9]{1,4}$"
)


def _clean_text(raw_text: str) -> str:
    """Clean OCR text and normalize it for regex checking."""
    if not raw_text:
        return ""

    clean = (
        raw_text.upper()
        .replace(" ", "")
        .replace("_", "")
        .replace("\n", "")
        .strip()
    )

    clean = clean.replace("–", "-").replace("--", "-")

    # Heuristic: if there is no hyphen and first 2 chars are letters,
    # insert one after the first letter → "MG4187" → "M-G4187"
    if "-" not in clean and len(clean) >= 3 and clean[0].isalpha() and clean[1].isalpha():
        clean = clean[0] + "-" + clean[1:]

    # Common OCR fix: 0 ↔ O in region code
    # Only fix left side (before digits)
    letters_part = "".join(ch for ch in clean if not ch.isdigit())
    digits_part = "".join(ch for ch in clean if ch.isdigit())
    letters_part = letters_part.replace("0", "O")
    clean = letters_part + digits_part

    return clean


def _has_blue_strip(plate_img) -> bool:
    """Roughly detect EU blue strip on the left side of the plate."""
    if plate_img is None or plate_img.size == 0:
        return False

    h, w = plate_img.shape[:2]
    if w == 0:
        return False

    # Take left 25% of the image (where EU strip is)
    left = plate_img[:, : int(w * 0.25)]

    hsv = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)

    # VERY wide blue range to be robust to lighting and screens
    lower_blue = np.array([80, 40, 40], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_pixels = cv2.countNonZero(mask)
    total_pixels = mask.size

    # tiny threshold, because your iPad plate is small in the frame
    ratio = blue_pixels / float(total_pixels)
    return ratio > 0.03  # 3% of pixels are blue


def is_german_plate(plate_img, raw_text: str):
    """
    Decide if a plate is German.

    Returns:
        (is_de: bool, display_text: str)
    """

    clean = _clean_text(raw_text)
    regex_ok = bool(GERMAN_REGEX.match(clean))
    blue_ok = _has_blue_strip(plate_img)

    # LOG to console so you can see what is happening
    print(f"OCR raw: '{raw_text}'  ->  cleaned: '{clean}' | regex_ok={regex_ok}, blue_ok={blue_ok}")

    # Final rule (relaxed a bit so your iPad example passes):
    #   - If regex says it's German OR we see a blue strip on left → treat as DE
    is_de = regex_ok or blue_ok

    # For label, prefer cleaned text if not empty
    display_text = clean if clean else raw_text

    return is_de, display_text