import os
import sys
import subprocess
import glob

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
YOLO_DIR = os.path.join(PROJECT_ROOT, "yolov5")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Use the test images for inference demonstration
TEST_IMAGES = os.path.join(DATA_DIR, "test", "images")

# The folder name defined in train_german_lpr.py
RUN_NAME = "german_lpr_medium_run" 
BEST_WEIGHTS = os.path.join(YOLO_DIR, "runs", "train", RUN_NAME, "weights", "best.pt")

def delete_old_yolov8_models():
    """Searches for and deletes YOLOv8 model files in the project root."""
    print("\n🧹 CLEANUP: Looking for old YOLOv8 models to delete...")
    # Patterns to look for common YOLOv8 weights
    patterns = ["yolov8*.pt", "yolov8*.onnx"]
    found = False
    
    for pattern in patterns:
        # Search in project root
        files = glob.glob(os.path.join(PROJECT_ROOT, pattern))
        # Also search in scripts folder just in case
        files += glob.glob(os.path.join(SCRIPT_DIR, pattern))
        
        for f in files:
            try:
                os.remove(f)
                print(f"   ❌ Deleted old model: {f}")
                found = True
            except OSError as e:
                print(f"   ⚠️ Could not delete {f}: {e}")
    
    if not found:
        print("   ✅ No old YOLOv8 models found to delete.")

def run_inference():
    """Runs YOLOv5 inference using the newly trained custom model."""
    if not os.path.exists(BEST_WEIGHTS):
        print(f"❌ ERROR: Could not find trained weights at: {BEST_WEIGHTS}")
        print("   Did the training finish successfully?")
        return

    print(f"\n🚀 STARTING INFERENCE WITH TUNED MODEL")
    print(f"   - Weights: {BEST_WEIGHTS}")
    print(f"   - Source:  {TEST_IMAGES}")
    
    detect_script = os.path.join(YOLO_DIR, "detect.py")
    
    cmd = [
        sys.executable, detect_script,
        "--weights", BEST_WEIGHTS,
        "--source", TEST_IMAGES,
        "--img", "640",
        "--conf", "0.25",       # Confidence threshold
        "--save-txt",           # Save labels
        "--save-conf",          # Save confidence in labels
        "--name", "inference_results",
        "--exist-ok"
    ]
    
    try:
        subprocess.check_call(cmd)
        print("\n✅ INFERENCE COMPLETE!")
        print(f"   Check results in: {os.path.join(YOLO_DIR, 'runs', 'detect', 'inference_results')}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Inference failed with error code {e.returncode}")

if __name__ == "__main__":
    delete_old_yolov8_models()
    run_inference()