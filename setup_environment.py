import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def download_yolo_model():
    """Download YOLOv8 model if not present"""
    from ultralytics import YOLO
    
    model_path = Path("yolov8s.pt")
    if not model_path.exists():
        print("Downloading YOLOv8s model...")
        try:
            model = YOLO("yolov8s.pt")
            print("✅ YOLOv8s model downloaded successfully!")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False
    else:
        print("✅ YOLOv8s model already exists!")
    return True

def verify_dataset():
    """Verify dataset structure"""
    print("Verifying dataset structure...")
    
    required_paths = [
        "data/train/images",
        "data/train/labels", 
        "data/val/images",
        "data/val/labels",
        "data/test/images",
        "data/test/labels",
        "classes.txt",
        "yolo_params.yaml"
    ]
    
    for path in required_paths:
        if not Path(path).exists():
            print(f"❌ Missing required path: {path}")
            return False
    
    # Count files
    train_images = len(list(Path("data/train/images").glob("*.png")))
    train_labels = len(list(Path("data/train/labels").glob("*.txt")))
    val_images = len(list(Path("data/val/images").glob("*.png")))
    val_labels = len(list(Path("data/val/labels").glob("*.txt")))
    test_images = len(list(Path("data/test/images").glob("*.png")))
    test_labels = len(list(Path("data/test/labels").glob("*.txt")))
    
    print(f"📊 Dataset Statistics:")
    print(f"   Train: {train_images} images, {train_labels} labels")
    print(f"   Val: {val_images} images, {val_labels} labels") 
    print(f"   Test: {test_images} images, {test_labels} labels")
    
    if train_images != train_labels or val_images != val_labels or test_images != test_labels:
        print("❌ Mismatch between images and labels!")
        return False
    
    print("✅ Dataset structure verified!")
    return True

def main():
    print("🚀 Setting up YOLO training environment...")
    
    # Install requirements
    if not install_requirements():
        return
    
    # Download model
    if not download_yolo_model():
        return
    
    # Verify dataset
    if not verify_dataset():
        return
    
    print("🎉 Environment setup complete! Ready to train.")

if __name__ == "__main__":
    main() 