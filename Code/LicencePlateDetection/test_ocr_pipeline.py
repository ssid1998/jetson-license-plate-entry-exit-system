import argparse
import glob
import os
import sys

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "jetson_app"))

from ocr_pipeline import PlateOcrPipeline  # noqa: E402


def loadImages(folder):
    patternsList = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    fileList = []
    for pattern in patternsList:
        fileList.extend(glob.glob(os.path.join(folder, pattern)))
    return fileList


def main():
    argParser = argparse.ArgumentParser(description="Quick test for the plate OCR pipeline.")
    argParser.add_argument(
        "--dir",
        dest="imageDir",
        default=os.path.join("data", "sample_plates"),
        help="Folder containing plate crops (default: data/sample_plates)",
    )
    argParser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate debug images to jetson_app/debug",
    )
    parsedArgs = argParser.parse_args()

    if not os.path.isdir(parsedArgs.imageDir):
        print(f"No image directory found at {parsedArgs.imageDir}. Put plate crops there and rerun.")
        return

    imagePaths = loadImages(parsedArgs.imageDir)
    if not imagePaths:
        print(f"No images found in {parsedArgs.imageDir}. Add plate crops to test OCR.")
        return

    pipeline = PlateOcrPipeline(debug=parsedArgs.debug)

    print(f"Running OCR on {len(imagePaths)} images from {parsedArgs.imageDir}")
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        if image is None:
            print(f"Could not read image: {imagePath}")
            continue

        text, conf, debugInfo = pipeline.readPlate(image)
        pipeline.updateTemporalVote(text, conf)
        smoothText, smoothConf = pipeline.getTemporalConsensus()

        displayText = smoothText or text
        displayConf = smoothConf if smoothText else conf
        infoText = f"{displayText} (conf={displayConf:.2f})" if displayText else "no text"

        print(f"{os.path.basename(imagePath)} -> {infoText}")
        if pipeline.debug:
            print(f"  Debug: {debugInfo}")


if __name__ == "__main__":
    main()
