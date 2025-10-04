#!/usr/bin/env python3
"""
Setup script for YOLOv8 Space Station Object Detection
Installs dependencies and prepares environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_conda_environment():
    """Create and activate conda environment."""
    print("🚀 Setting up YOLOv8 Space Station Object Detection Environment")
    
    # Check if conda is available
    if not run_command("conda --version", "Checking conda availability"):
        print("❌ Conda not found. Please install Anaconda or Miniconda first.")
        return False
    
    # Create environment
    if not run_command("conda env create -f environment.yml", "Creating conda environment"):
        print("❌ Failed to create conda environment")
        return False
    
    print("✅ Conda environment 'EDU' created successfully")
    return True

def install_additional_dependencies():
    """Install additional dependencies not in environment.yml."""
    additional_packages = [
        "tensorboard",
        "wandb",
        "albumentations"
    ]
    
    for package in additional_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Warning: Failed to install {package}")
    
    return True

def create_project_structure():
    """Create necessary project directories."""
    directories = [
        "dataset/train/images",
        "dataset/train/labels", 
        "dataset/val/images",
        "dataset/val/labels",
        "dataset/test/images",
        "dataset/test/labels",
        "runs/train",
        "runs/val",
        "logs",
        "models",
        "results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def download_yolov8_models():
    """Download pre-trained YOLOv8 models."""
    models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    
    for model in models:
        model_path = Path(f"models/{model}")
        if not model_path.exists():
            print(f"📥 Downloading {model}...")
            # This will be handled by ultralytics when needed
            print(f"✅ {model} will be downloaded automatically when training starts")
        else:
            print(f"✅ {model} already exists")
    
    return True

def create_sample_data_yaml():
    """Create a sample data.yaml file."""
    data_yaml_content = """# YOLOv8 Dataset Configuration
# Space Station Object Detection Dataset

path: ./dataset  # Dataset root directory
train: train/images  # Train images (relative to 'path')
val: val/images      # Val images (relative to 'path')
test: test/images    # Test images (relative to 'path')

# Classes
names:
  0: Toolbox
  1: Oxygen Tank
  2: Fire Extinguisher
"""
    
    with open("data.yaml", "w") as f:
        f.write(data_yaml_content)
    
    print("✅ Created data.yaml configuration file")
    return True

def test_installation():
    """Test if the installation is working correctly."""
    print("🧪 Testing installation...")
    
    test_script = """
import sys
import torch
import ultralytics
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

print("✅ All required packages imported successfully")
print(f"PyTorch version: {torch.__version__}")
print(f"Ultralytics version: {ultralytics.__version__}")
print(f"OpenCV version: {cv2.__version__}")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation test failed: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("🚀 YOLOv8 Space Station Object Detection Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create conda environment
    if not create_conda_environment():
        return False
    
    # Install additional dependencies
    install_additional_dependencies()
    
    # Create project structure
    create_project_structure()
    
    # Download models
    download_yolov8_models()
    
    # Create sample data.yaml
    create_sample_data_yaml()
    
    # Test installation
    if test_installation():
        print("\n" + "=" * 60)
        print("🎉 Setup completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Activate the environment: conda activate EDU")
        print("2. Prepare your dataset in the dataset/ directory")
        print("3. Update config.yaml with your settings")
        print("4. Run training: python train.py")
        print("5. Run evaluation: python predict.py --model runs/train/yolov8_training/weights/best.pt")
        print("6. Launch web app: streamlit run app.py")
        print("\nFor more information, see README.md")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 