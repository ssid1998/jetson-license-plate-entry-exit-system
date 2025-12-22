import flet as ft
import numpy as np
import cv2
import threading
import time
import base64
import os
import torch
from datetime import datetime

from ocr_pipeline import PlateOcrPipeline
from db import DatabaseManager
from country_check import is_german_plate
from billing_logic import calculate_parking_fee

class ParkingDashboard:
    def __init__(self):
        self.running = True
        self.latest_frame_b64 = None
        self.latest_plate_crop_b64 = None
        self.last_plate_time = 0
        self.plate_cooldowns = {}
        self.plate_buffer = []
        self.last_capture_time = 0
        self.detected_plate = "Waiting..."
        self.status_message = "Ready"
        self.price_display = "€ 0.00"
        self.duration_display = "0 min"
        self.status_color = ft.Colors.BLUE
        
        # Initialize Logic Components
        self.db = DatabaseManager()
        self.init_ai()

    def init_ai(self):
        print("Loading AI Models...")
        # Define paths for YOLOv5
        base_path = os.path.dirname(os.path.abspath(__file__))
        yolo_repo = os.path.join(base_path, "..", "yolov5")
        
        # Path to the trained weights from train_german_lpr.py
        weights_path = os.path.join(
            yolo_repo,
            "runs",
            "train",
            "german_lpr_medium_run",
            "weights",
            "best.pt"
        )
        print(f"Loading YOLOv5 model from: {weights_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load YOLOv5 using torch.hub from local repo
        self.yoloModel = torch.hub.load(yolo_repo, 'custom', path=weights_path, source='local')
        self.yoloModel.to(device)
        
        self.ocrPipeline = PlateOcrPipeline(
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ", # Added hyphen and space
            confThreshold=0.4,   # Lowered slightly more to ensure we get a reading
            euMaskRatio=0.15,
            debug=False,
            useGpu=torch.cuda.is_available()
        )
        self.device = device

    def process_camera(self, page: ft.Page, img_control: ft.Image):
        # Try external camera (index 1) first, fallback to default (index 0)
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("External camera (index 1) not found. Switching to default (index 0).")
            cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print("Error: Camera not found")
            return

        last_process_time = 0
        process_interval = 0.2  # Process AI every 200ms to save resources
        prev_gray = None
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Motion Detection Optimization
            # We only run the heavy AI if something is moving or if we recently saw a car
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if prev_gray is None:
                prev_gray = gray
                continue
                
            frameDelta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = np.sum(thresh)
            prev_gray = gray

            now = time.time()
            
            # Run AI processing periodically ONLY if motion detected or recently tracking
            if (motion_score > 5000 or (now - self.last_plate_time < 3.0)):
                if now - last_process_time > process_interval:
                    self.run_detection(frame)
                    last_process_time = now
            
            self.update_ui(page, img_control, frame)

            # Small sleep to prevent CPU hogging
            time.sleep(0.01)

        cap.release()

    def run_detection(self, frame):
        # Configure YOLOv5 thresholds
        self.yoloModel.conf = 0.2
        self.yoloModel.iou = 0.5
        
        # Run Inference
        results = self.yoloModel(frame, size=640)
        detections = results.xyxy[0].cpu().numpy() # [x1, y1, x2, y2, conf, cls]
        
        found_plate = False
        
        # Reset buffer if no plate seen for a while (1 second timeout)
        if len(detections) == 0:
            if time.time() - self.last_capture_time > 1.0:
                self.plate_buffer = []
            return
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Add Padding (10% width, 15% height) to ensure characters aren't cut off
            h, w = frame.shape[:2]
            pad_w = int((x2 - x1) * 0.10)
            pad_h = int((y2 - y1) * 0.15)
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)

            plateCrop = frame[y1:y2, x1:x2]
            if plateCrop.size == 0:
                continue
            
            found_plate = True
            self.last_plate_time = time.time()
            self.last_capture_time = time.time()
            
            # Calculate sharpness (Variance of Laplacian)
            gray_crop = cv2.cvtColor(plateCrop, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
            
            self.plate_buffer.append((plateCrop, sharpness))
            

            # Draw bounding box (visual feedback only)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if len(self.plate_buffer) >= 5:
                # Pick the best image (highest sharpness)
                best_crop = max(self.plate_buffer, key=lambda x: x[1])[0]
                
                # Update the "Screenshot" in UI
                _, buffer = cv2.imencode('.jpg', best_crop)
                self.latest_plate_crop_b64 = base64.b64encode(buffer).decode('utf-8')

                # Run OCR on the best crop
                plateText, plateConf, _ = self.ocrPipeline.readPlate(best_crop)
                
                # Country Check
                isGerman, checkedText = is_german_plate(best_crop, plateText)
                displayText = checkedText if checkedText else plateText
                
                # Update the main display text immediately with the best result
                self.detected_plate = displayText if displayText else "Unreadable"
                
                # Log unreadable images for debugging
                if not displayText:
                    ts = int(time.time() * 1000)
                    fail_dir = os.path.join(os.path.dirname(__file__), "captures", "unreadable")
                    os.makedirs(fail_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(fail_dir, f"fail_{ts}.jpg"), best_crop)

                if displayText and isGerman:
                    # Cooldown Logic: Prevent rapid re-triggering for the same plate
                    now = time.time()
                    last_processed = self.plate_cooldowns.get(displayText, 0)
                    
                    if (now - last_processed) > 10.0:
                        self.handle_plate_logic(displayText, plateConf, frame)
                        self.plate_cooldowns[displayText] = now
                
                # Clear buffer to restart process or wait for next car
                self.plate_buffer = []
                
                break # Only handle one plate at a time

    def handle_plate_logic(self, plate_text, confidence, frame):
        # 1. Log current detection
        # We don't want to spam DB, so maybe check if we logged this recently?
        # For this dashboard, we assume we log it to track "last seen"
        # But to avoid DB bloat, let's just read first detection.
        
        # Check if we have seen this car today
        first_seen = self.db.get_first_detection_today(plate_text)
        
        now = datetime.now()
        
        if not first_seen:
            # First time seeing it today -> Entry
            self.db.log_detection(plate_text, confidence)
            self.detected_plate = plate_text
            self.status_message = "WELCOME"
            self.status_color = ft.Colors.GREEN
            self.price_display = "Free"
            self.duration_display = "Just Arrived"
        else:
            # Seen before -> Exit / Update
            # Log this new sighting so we know it's still here (optional, but good for history)
            # self.db.log_detection(plate_text, confidence) 
            
            self.detected_plate = plate_text
            
            duration = now - first_seen
            minutes = int(duration.total_seconds() / 60)
            fee = calculate_parking_fee(first_seen, now)
            
            self.duration_display = f"{minutes} min"
            self.price_display = f"€ {fee:.2f}"
            
            if fee == 0:
                self.status_message = "FREE PARKING"
                self.status_color = ft.Colors.GREEN
            else:
                self.status_message = "GOODBYE - PLEASE PAY"
                self.status_color = ft.Colors.RED

    def update_ui(self, page, img_control, frame):
        # Convert frame to base64 for Flet
        # Resize for UI performance (display at 640x480, but process at 720p)
        display_frame = cv2.resize(frame, (640, 480))
        _, buffer = cv2.imencode('.jpg', display_frame)
        b64_img = base64.b64encode(buffer).decode('utf-8')
        
        img_control.src_base64 = b64_img
        img_control.update()
        
        if self.latest_plate_crop_b64:
            self.img_plate_crop.src_base64 = self.latest_plate_crop_b64
            self.img_plate_crop.update()
        
        # Update text fields
        self.txt_plate.value = self.detected_plate
        self.txt_status.value = self.status_message
        self.txt_status.color = self.status_color
        self.txt_price.value = self.price_display
        self.txt_duration.value = self.duration_display
        self.txt_time.value = datetime.now().strftime("%H:%M:%S")
        
        page.update()

    def on_manual_submit(self, e):
        text = self.txt_manual_input.value.upper().strip()
        if text:
            self.detected_plate = text
            self.handle_plate_logic(text, 1.0, None)
            self.txt_manual_input.value = ""
            self.page.update()

    def build(self, page: ft.Page):
        page.title = "Smart Parking Dashboard"
        page.theme_mode = ft.ThemeMode.DARK
        page.window_width = 1000
        page.window_height = 800

        # UI Components
        self.txt_time = ft.Text(value="--:--:--", size=20, weight=ft.FontWeight.BOLD)
        self.txt_plate = ft.Text(value="---", size=50, weight=ft.FontWeight.BOLD)
        self.txt_status = ft.Text(value="Ready", size=30, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE)
        self.txt_price = ft.Text(value="€ 0.00", size=40, weight=ft.FontWeight.BOLD, color=ft.Colors.AMBER)
        self.txt_duration = ft.Text(value="0 min", size=20)
        
        img_control = ft.Image(src_base64="", width=640, height=480, fit=ft.ImageFit.CONTAIN)
        self.img_plate_crop = ft.Image(src_base64="", width=200, height=80, fit=ft.ImageFit.CONTAIN)
        
        # Manual Override Controls
        self.txt_manual_input = ft.TextField(label="Manual Override", width=200, text_size=16)
        self.btn_manual_submit = ft.ElevatedButton("Submit", on_click=self.on_manual_submit)

        # Layout
        header = ft.Row([
            ft.Icon(ft.Icons.DIRECTIONS_CAR, size=40),
            ft.Text("Smart Parking System", size=30, weight=ft.FontWeight.BOLD),
            ft.Container(expand=True),
            self.txt_time
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        info_panel = ft.Column([
            ft.Text("License Plate:", size=15, color=ft.Colors.GREY),
            self.img_plate_crop,
            self.txt_plate,
            ft.Divider(),
            self.txt_status,
            ft.Divider(),
            ft.Text("Duration:", size=15, color=ft.Colors.GREY),
            self.txt_duration,
            ft.Text("Amount Due:", size=15, color=ft.Colors.GREY),
            self.txt_price,
            ft.Divider(),
            ft.Row([self.txt_manual_input, self.btn_manual_submit], alignment=ft.MainAxisAlignment.CENTER)
        ], spacing=10, alignment=ft.MainAxisAlignment.CENTER)

        main_row = ft.Row([
            ft.Container(content=img_control, border=ft.border.all(2, ft.Colors.GREY_800), border_radius=10),
            ft.Container(content=info_panel, padding=20, expand=True, bgcolor=ft.Colors.GREY_900, border_radius=10)
        ], expand=True)

        page.add(header, ft.Divider(), main_row)

        # Start Camera Thread
        t = threading.Thread(target=self.process_camera, args=(page, img_control), daemon=True)
        t.start()
        self.page = page

def main(page: ft.Page):
    app = ParkingDashboard()
    app.build(page)

if __name__ == "__main__":
    ft.app(target=main)