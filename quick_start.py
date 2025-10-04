#!/usr/bin/env python3
"""
Quick Start Script for YOLOv8 Space Station Object Detection
Provides an easy way to get started with the project
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """Print project banner."""
    print("=" * 60)
    print("🚀 YOLOv8 Space Station Object Detection")
    print("=" * 60)
    print("Detecting Toolbox, Oxygen Tank, and Fire Extinguisher")
    print("Target: ≥90% mAP@0.5 accuracy")
    print("=" * 60)

def check_environment():
    """Check if the environment is properly set up."""
    print("🔍 Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check if conda environment is activated
    if 'EDU' not in os.environ.get('CONDA_DEFAULT_ENV', ''):
        print("⚠️  Warning: EDU conda environment not activated")
        print("   Run: conda activate EDU")
    
    # Check required files
    required_files = ['config.yaml', 'train.py', 'predict.py', 'app.py']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ Environment check passed")
    return True

def setup_dataset():
    """Set up dataset structure."""
    print("📁 Setting up dataset structure...")
    
    dataset_dirs = [
        "dataset/train/images",
        "dataset/train/labels",
        "dataset/val/images", 
        "dataset/val/labels",
        "dataset/test/images",
        "dataset/test/labels"
    ]
    
    for dir_path in dataset_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")
    
    print("📝 Dataset structure ready!")
    print("   Place your images and labels in the dataset/ directory")
    print("   Images: dataset/{train,val,test}/images/")
    print("   Labels: dataset/{train,val,test}/labels/")

def run_training_demo():
    """Run a quick training demo with sample data."""
    print("🎯 Running training demo...")
    
    # Check if dataset has data
    train_images = list(Path("dataset/train/images").glob("*.jpg")) + list(Path("dataset/train/images").glob("*.png"))
    
    if not train_images:
        print("⚠️  No training images found in dataset/train/images/")
        print("   Please add your dataset before training")
        return False
    
    print(f"📊 Found {len(train_images)} training images")
    
    # Run training with nano model for quick demo
    cmd = [
        sys.executable, "train.py",
        "--config", "config.yaml",
        "--model-size", "n"
    ]
    
    print("🚀 Starting training...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False

def run_evaluation_demo():
    """Run evaluation on trained model."""
    print("📊 Running evaluation demo...")
    
    model_path = "runs/train/yolov8_training/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Trained model not found: {model_path}")
        print("   Please run training first")
        return False
    
    cmd = [
        sys.executable, "predict.py",
        "--model", model_path,
        "--config", "config.yaml"
    ]
    
    print("🔍 Starting evaluation...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def launch_web_app():
    """Launch the Streamlit web application."""
    print("🌐 Launching web application...")
    
    cmd = ["streamlit", "run", "app.py"]
    
    print("🚀 Starting Streamlit app...")
    print(f"   Command: {' '.join(cmd)}")
    print("   The app will open in your browser")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch web app: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Web app stopped")
        return True

def show_next_steps():
    """Show next steps for the user."""
    print("\n" + "=" * 60)
    print("🎯 Next Steps")
    print("=" * 60)
    print("1. 📁 Prepare your dataset:")
    print("   - Add images to dataset/{train,val,test}/images/")
    print("   - Add YOLO labels to dataset/{train,val,test}/labels/")
    print("   - Run: python data_utils.py --action analyze --dataset-dir dataset")
    print()
    print("2. 🎯 Train your model:")
    print("   - Quick: python train.py --config config.yaml --model-size n")
    print("   - Better: python train.py --config config.yaml --model-size m")
    print("   - Best: python train.py --config config.yaml --model-size l")
    print()
    print("3. 📊 Evaluate your model:")
    print("   - python predict.py --model runs/train/yolov8_training/weights/best.pt")
    print()
    print("4. 🌐 Use the web app:")
    print("   - streamlit run app.py")
    print()
    print("5. 🔧 Optimize for ≥90% mAP@0.5:")
    print("   - Use larger models (YOLOv8s, YOLOv8m, YOLOv8l)")
    print("   - Increase training epochs")
    print("   - Add more diverse data")
    print("   - Check optimization_recommendations.txt")
    print()
    print("📚 For detailed instructions, see README.md")

def main():
    """Main quick start function."""
    parser = argparse.ArgumentParser(description='Quick Start for YOLOv8 Space Station Object Detection')
    parser.add_argument('--action', type=str, choices=['setup', 'train', 'eval', 'web', 'all'], 
                       default='setup', help='Action to perform')
    parser.add_argument('--skip-checks', action='store_true', help='Skip environment checks')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check environment
    if not args.skip_checks and not check_environment():
        print("❌ Environment check failed. Please fix the issues above.")
        return False
    
    if args.action == 'setup':
        setup_dataset()
        show_next_steps()
    
    elif args.action == 'train':
        setup_dataset()
        if run_training_demo():
            show_next_steps()
    
    elif args.action == 'eval':
        if run_evaluation_demo():
            show_next_steps()
    
    elif args.action == 'web':
        launch_web_app()
    
    elif args.action == 'all':
        setup_dataset()
        if run_training_demo():
            if run_evaluation_demo():
                launch_web_app()
    
    print("\n🎉 Quick start completed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 