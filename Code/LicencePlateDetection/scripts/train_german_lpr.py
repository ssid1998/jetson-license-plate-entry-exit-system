import os
import sys
import shutil
import subprocess
import torch
import yaml

# --- 1. SYSTEM CONFIGURATION (CRITICAL FOR WINDOWS) ---
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Prevents common library crashes

# --- 2. PROJECT SETTINGS ---
# Robustly find the project root (LicencePlateDetection folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Point to your existing folder (yolov5-master)
YOLO_DIR = os.path.join(PROJECT_ROOT, "yolov5") 

# MODEL SETTINGS
# 'yolov5m.pt' is the Medium model. Much smarter than 's' or 'n', but requires more GPU RAM.
# If you run out of memory, change this back to "yolov5s.pt".
MODEL_WEIGHTS = "yolov5m.pt" 

EPOCHS = 100            # Adjusted to 100 as 300 is often overkill for LPR
BATCH_SIZE = 8          # Lowered slightly to fit the larger 'Medium' model in memory
IMG_SIZE = 640          # Standard resolution
YAML_FILENAME = "german_plates.yaml"
RUN_NAME = "german_lpr_medium_run" # Name of the folder where results will be saved

def create_data_yaml(yaml_path):
    """
    Generates the dataset configuration file dynamically to fix path issues.
    """
    data_dir = os.path.join(PROJECT_ROOT, "data")
    
    # Check for 'val' or 'valid' folder
    val_folder = "val"
    if os.path.exists(os.path.join(data_dir, "valid")):
        val_folder = "valid"

    print(f"🔍 Creating dataset config at: {yaml_path}")
    print(f"   - Data Directory: {data_dir}")
    
    yaml_content = f"""
path: {data_dir.replace(os.sep, '/')}  # dataset root dir
train: train/images
val: {val_folder}/images
test: test/images

nc: 1
names: ['license_plate']
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print("✅ german_plates.yaml created successfully.")

def setup_environment():
    """
    Prepares the YOLOv5 environment, fixes missing files, and ensures requirements.
    """
    print(f"\n🔍 Checking configuration...")
    print(f"   - Project Root: {PROJECT_ROOT}")
    print(f"   - YOLO Directory: {YOLO_DIR}")

    if not os.path.exists(YOLO_DIR):
        print(f"❌ ERROR: Could not find {YOLO_DIR}")
        print("   Please ensure you have extracted the 'yolov5-master' folder correctly.")
        sys.exit(1)

    # --- FIX MISSING HYPERPARAMETER FILE ---
    # Newer YOLO versions removed 'hyp.scratch-med.yaml'. We must create it manually.
    hyps_folder = os.path.join(YOLO_DIR, "data", "hyps")
    source_hyp = os.path.join(hyps_folder, "hyp.scratch-low.yaml") # Use low as base
    target_hyp = os.path.join(hyps_folder, "hyp.scratch-med.yaml")

    if os.path.exists(source_hyp) and not os.path.exists(target_hyp):
        print("🔧 Creating missing config file (hyp.scratch-med.yaml)...")
        shutil.copy(source_hyp, target_hyp)
        print("✅ Config fixed.")
    
    # --- CREATE DATASET YAML ---
    yaml_path = os.path.join(PROJECT_ROOT, YAML_FILENAME)
    create_data_yaml(yaml_path)

    # --- CHECK GPU ---
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"✅ NVIDIA CUDA GPU Detected: {count} device(s)")
        for i in range(count):
            print(f"   - Device {i}: {torch.cuda.get_device_name(i)}")
        print("   -> Using Device 0 for training (Best Performance)")
    else:
        print("⚠️ WARNING: No GPU detected. Training will be very slow on CPU.")
        print("   If you have an NVIDIA GPU, ensure CUDA drivers and PyTorch-CUDA are installed.")

def check_and_resume_logic(cmd, run_dir):
    """
    Checks if a previous run exists. If so, updates the config to the new EPOCH count
    and modifies the command to resume instead of start over.
    """
    last_pt = os.path.join(run_dir, "weights", "last.pt")
    opt_yaml = os.path.join(run_dir, "opt.yaml")

    if os.path.exists(last_pt) and os.path.exists(opt_yaml):
        print(f"\n🔄 FOUND INTERRUPTED TRAINING AT: {last_pt}")
        print(f"   - Modifying {opt_yaml} to set epochs={EPOCHS}...")
        
        # 1. Update the saved options file to reflect the new epoch count
        try:
            with open(opt_yaml, 'r') as f:
                opt_data = yaml.safe_load(f)
            
            opt_data['epochs'] = EPOCHS
            
            with open(opt_yaml, 'w') as f:
                yaml.safe_dump(opt_data, f)
            print("   - Config updated successfully.")
            
            # 2. Change command to resume
            # Remove arguments that are ignored during resume or cause conflicts
            print("   - Switching to RESUME mode (starts from last saved epoch).")
            return [sys.executable, cmd[1], "--resume", last_pt]
            
        except Exception as e:
            print(f"⚠️ Failed to update resume config: {e}. Starting fresh/finetuning instead.")
    
    return cmd

def run_training():
    """
    Executes the training command with professional flags.
    """
    print("\n🚀 INITIALIZING PROFESSIONAL TRAINING SESSION...")
    
    # ANALYSIS OF LOSS FUNCTION FOR MAXIMUM ACCURACY
    print("ℹ️  ACCURACY & LOSS FUNCTION ANALYSIS:")
    print("   - Model: YOLOv5m (Medium) selected for better accuracy than Nano/Small.")
    print("   - Loss Function: YOLOv5 uses CIoU (Complete IoU) for bounding boxes and BCEWithLogitsLoss for classification.")
    print("   - Optimization: This combination maximizes mAP (Mean Average Precision) for object detection.")

    train_script = os.path.join(YOLO_DIR, "train.py")
    hyp_file = os.path.join(YOLO_DIR, "data", "hyps", "hyp.scratch-med.yaml")
    
    data_yaml_path = os.path.join(PROJECT_ROOT, YAML_FILENAME)
    run_dir = os.path.join(YOLO_DIR, "runs", "train", RUN_NAME)
    
    # The Command
    cmd = [
        sys.executable, train_script,
        "--img", str(IMG_SIZE),
        "--batch", str(BATCH_SIZE),
        "--epochs", str(EPOCHS),
        "--data", data_yaml_path,      # Absolute path to config
        "--weights", MODEL_WEIGHTS,    # yolov5m.pt
        "--hyp", hyp_file,             # The fixed config
        "--name", RUN_NAME,            # Organized folder name
        "--exist-ok",                  # Don't crash if folder exists
        "--patience", "50",            # Stop if no improvement for 50 epochs (Save time)
        "--workers", "2",              # Safe setting for Windows to prevent loading crashes
        "--device", "0"                # Force use of the primary NVIDIA GPU
    ]

    # Check if we should resume from an existing run
    cmd = check_and_resume_logic(cmd, run_dir)

    print(f"   - Model: {MODEL_WEIGHTS}")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print("-------------------------------------------------------")
    
    try:
        # Run the command and show output in real-time
        subprocess.check_call(cmd)
        print("\n✅ TRAINING COMPLETE!")
        print(f"   Results saved to: {os.path.join(YOLO_DIR, 'runs', 'train', RUN_NAME)}")
        
        # --- RUN TEST EVALUATION ---
        print("\n📊 RUNNING EVALUATION ON TEST SET (MAXIMUM ACCURACY CHECK)...")
        val_script = os.path.join(YOLO_DIR, "val.py")
        best_weights = os.path.join(YOLO_DIR, "runs", "train", RUN_NAME, "weights", "best.pt")
        
        if os.path.exists(val_script) and os.path.exists(best_weights):
            test_cmd = [
                sys.executable, val_script,
                "--weights", best_weights,
                "--data", data_yaml_path,
                "--task", "test",
                "--img", str(IMG_SIZE),
                "--name", f"{RUN_NAME}_test"
            ]
            subprocess.check_call(test_cmd)
            print("✅ Test evaluation finished.")
        else:
            print("⚠️ Skipping test evaluation (val.py or best.pt not found).")
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        print("   Check the error message above for details.")

if __name__ == "__main__":
    setup_environment()
    run_training()