import os
import re
from collections import deque
from typing import List, Tuple

import cv2
import easyocr
import numpy as np


class PlateOcrPipeline:
    """
    Pipeline to run EasyOCR on German license plates with preprocessing,
    validation, and simple temporal smoothing.
    """

    _reader = None

    def __init__(self, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", confThreshold=0.6, euMaskRatio=0.15, debug=False, useGpu=True):
        self.allowlist = allowlist
        self.confThreshold = confThreshold
        self.euMaskRatio = euMaskRatio
        self.debug = debug
        self.useGpu = useGpu

        self.resizeWidth = 384
        self.strictValidation = True
        self.debugDir = os.path.join(os.path.dirname(__file__), "debug")
        self.temporalReads = deque(maxlen=5)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self._ensureReader()

    def _ensureReader(self):
        if PlateOcrPipeline._reader is None:
            try:
                PlateOcrPipeline._reader = easyocr.Reader(
                    ["en"], gpu=self.useGpu, verbose=False, download_enabled=False
                )
            except Exception:
                PlateOcrPipeline._reader = easyocr.Reader(
                    ["en"], gpu=False, verbose=False, download_enabled=False
                )
        self.reader = PlateOcrPipeline._reader

    def _saveDebugImage(self, name: str, img):
        if not self.debug or img is None or img.size == 0:
            return
        os.makedirs(self.debugDir, exist_ok=True)
        savePath = os.path.join(self.debugDir, name)
        cv2.imwrite(savePath, img)

    def _maskEuBand(self, plateBgr):
        if plateBgr is None or plateBgr.size == 0:
            return plateBgr

        height, width = plateBgr.shape[:2]
        if width == 0:
            return plateBgr

        startX = int(self.euMaskRatio * width)
        if startX >= width:
            return plateBgr

        masked = plateBgr[:, startX:]
        self._saveDebugImage("plate_masked.jpg", masked)
        return masked

    def preprocessPlate(self, plateBgr) -> List[Tuple[str, np.ndarray]]:
        processedVariants: List[Tuple[str, np.ndarray]] = []

        if plateBgr is None or plateBgr.size == 0:
            return processedVariants

        grayImg = cv2.cvtColor(plateBgr, cv2.COLOR_BGR2GRAY)
        height, width = grayImg.shape[:2]
        if height == 0 or width == 0:
            return processedVariants

        targetW = self.resizeWidth
        targetH = max(1, int(height * (targetW / float(width))))
        grayImg = cv2.resize(grayImg, (targetW, targetH), interpolation=cv2.INTER_CUBIC)
        self._saveDebugImage("plate_gray.jpg", grayImg)

        claheImg = self.clahe.apply(grayImg)
        blurredImg = cv2.GaussianBlur(claheImg, (5, 5), 0)
        sharpenedImg = cv2.addWeighted(claheImg, 1.5, blurredImg, -0.5, 0)
        self._saveDebugImage("plate_sharpened.jpg", sharpenedImg)

        adaptive = cv2.adaptiveThreshold(
            sharpenedImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 7
        )
        _, otsu = cv2.threshold(sharpenedImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        adaptiveMorph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, morphKernel)
        otsuMorph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, morphKernel)

        processedVariants.append(("adaptive", adaptiveMorph))
        processedVariants.append(("otsu", otsuMorph))
        processedVariants.append(("adaptive_no_morph", adaptive))
        processedVariants.append(("otsu_no_morph", otsu))

        for idx, (name, img) in enumerate(processedVariants):
            self._saveDebugImage(f"plate_pre_{idx+1}_{name}.jpg", img)

        return processedVariants

    def runEasyOcr(self, plateImg) -> Tuple[str, float]:
        if plateImg is None or plateImg.size == 0:
            return "", 0.0

        try:
            ocrResults = self.reader.readtext(
                plateImg,
                detail=1,
                paragraph=False,
                allowlist=self.allowlist,
                text_threshold=0.4,
                low_text=0.3,
            )
        except Exception:
            return "", 0.0

        if not ocrResults:
            return "", 0.0

        bestLine = max(ocrResults, key=lambda r: r[2] if len(r) > 2 else 0.0)
        text = bestLine[1] if len(bestLine) > 1 else ""
        conf = float(bestLine[2]) if len(bestLine) > 2 and bestLine[2] is not None else 0.0
        return text, conf

    def normalizeText(self, text: str) -> str:
        if not text:
            return ""
        return "".join(ch for ch in text.upper() if ch.isalnum())

    def postCorrect(self, text: str) -> str:
        if not text:
            return ""

        chars = list(text)
        firstDigit = next((i for i, c in enumerate(chars) if c.isdigit()), len(chars))

        for i in range(firstDigit):
            if chars[i] == "0":
                chars[i] = "O"
            elif chars[i] == "1":
                chars[i] = "I"
            elif chars[i] == "5":
                chars[i] = "S"
            elif chars[i] == "8":
                chars[i] = "B"

        for i in range(firstDigit, len(chars)):
            if chars[i] == "O":
                chars[i] = "0"
            elif chars[i] == "I":
                chars[i] = "1"
            elif chars[i] == "B":
                chars[i] = "8"
            elif chars[i] == "S":
                chars[i] = "5"
            elif chars[i] == "Z":
                chars[i] = "2"
            elif chars[i] == "G":
                chars[i] = "6"

        return "".join(chars)

    def validateGermanPlate(self, text: str) -> bool:
        if not text:
            return False

        if not re.match(r"^[A-Z]{1,3}[A-Z]{1,2}[0-9]{1,4}$", text):
            return False

        lettersCount = sum(1 for c in text if c.isalpha())
        digitsCount = sum(1 for c in text if c.isdigit())
        if self.strictValidation and (digitsCount < 1 or lettersCount < 2):
            return False

        return True

    def _logDebug(self, message: str):
        if self.debug:
            print(message)

    def readPlate(self, plateBgr):
        debugInfo = {"variants": [], "best": {}}

        if plateBgr is None or plateBgr.size == 0:
            self._logDebug("Empty plate crop, skipping OCR.")
            return "", 0.0, debugInfo

        self._saveDebugImage("plate_raw.jpg", plateBgr)
        maskedCrop = self._maskEuBand(plateBgr)

        variants = self.preprocessPlate(maskedCrop)
        bestText, bestConf, bestVariant = "", 0.0, ""

        for name, variant in variants:
            rawText, conf = self.runEasyOcr(variant)
            normalized = self.normalizeText(rawText)
            corrected = self.postCorrect(normalized)
            isValid = self.validateGermanPlate(corrected)

            debugEntry = {
                "variant": name,
                "raw_text": rawText,
                "normalized": normalized,
                "corrected": corrected,
                "confidence": conf,
                "regex_ok": isValid,
            }
            debugInfo["variants"].append(debugEntry)

            self._logDebug(
                f"[{name}] raw='{rawText}' norm='{normalized}' corrected='{corrected}' conf={conf:.3f} valid={isValid}"
            )

            if isValid and conf >= self.confThreshold:
                bestText, bestConf, bestVariant = corrected, conf, name
                break

            if conf > bestConf:
                bestText, bestConf, bestVariant = corrected, conf, name

        debugInfo["best"] = {"text": bestText, "confidence": bestConf, "variant": bestVariant}

        return bestText, bestConf, debugInfo

    def updateTemporalVote(self, text: str, conf: float):
        if not text:
            return
        self.temporalReads.append((text, conf))

    def getTemporalConsensus(self) -> Tuple[str, float]:
        if not self.temporalReads:
            return "", 0.0

        voteMap = {}
        for candidate, confidence in self.temporalReads:
            if not candidate:
                continue
            if candidate not in voteMap:
                voteMap[candidate] = []
            voteMap[candidate].append(confidence)

        if not voteMap:
            return "", 0.0

        bestText = max(voteMap.items(), key=lambda item: (len(item[1]), np.mean(item[1])))[0]
        bestConf = float(np.mean(voteMap[bestText]))
        return bestText, bestConf
