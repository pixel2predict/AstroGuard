🚀AstroGuard - YOLOv8 Space Station Object Detection

A comprehensive object detection pipeline leveraging **YOLOv8** has been developed to accurately detect and classify **Toolboxes, Oxygen Tanks, and Fire Extinguishers** within **space station environments**. To overcome the challenges of limited real-world annotated datasets, the system is trained and validated using high-fidelity **synthetic data generated from the Duality AI Falcon platform**. This integration of advanced simulation with state-of-the-art deep learning ensures robust performance, enabling reliable safety monitoring and resource tracking in critical space missions.

🎯 Project Goals

> Achieving ≥90% mAP@0.5 accuracy across target classes.
> Ensuring robust detection under varying lighting conditions typical of orbital environments.
> Handling occlusions and partial visibility of objects in cluttered station interiors.
> Delivering real-time inference capabilities suitable for mission-critical applications.
> Providing comprehensive evaluation and visualization tools for monitoring and performance assessment.

📋 Table of Contents

Features
Installation
Dataset Preparation
Training
Evaluation
Web Application
Project Structure
Configuration
Troubleshooting

✨ Features

🎯 Core Functionality

YOLOv8 Training Pipeline with hyperparameter optimization
Data Augmentation (mosaic, HSV, flip, rotation, scaling)
Comprehensive Evaluation (mAP@0.5, confusion matrix, per-class metrics)
Failure Case Analysis for model improvement
Real-time Web Application with Streamlit
Model Optimization Strategies for accuracy improvement

📊 Analytics & Visualization

Training curves and loss plots
Confusion matrix visualization
Per-class performance metrics
Dataset distribution analysis
Failure case documentation

🔧 Technical Features

Modular and clean code architecture
Comprehensive logging and error handling
GPU/CPU support with automatic device detection
Configurable hyperparameters via YAML
Model checkpointing and resume training

🚀 Installation

Prerequisites

Python 3.8+
Anaconda or Miniconda
CUDA-compatible GPU (recommended)
Quick Setup

Clone the repository:
git clone <repository-url>
cd space-station-object-detection
Run the setup script:
python setup.py
Activate the environment:
conda activate EDU
Manual Installation

Create conda environment:
conda env create -f environment.yml
conda activate EDU
Install additional dependencies:
pip install tensorboard wandb albumentations
Create project structure:
mkdir -p dataset/{train,val,test}/{images,labels}
mkdir -p runs/{train,val}
mkdir -p logs models results
📁 Dataset Preparation

Dataset Structure

dataset/

├── train/

│   ├── images/

│   └── labels/

├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
YOLO Format Labels

Each label file (.txt) should contain one line per object:

class_id x_center y_center width height
Where:

class_id: 0 (Toolbox), 1 (Oxygen Tank), 2 (Fire Extinguisher)
x_center, y_center, width, height: Normalized coordinates [0, 1]
Data Preparation Scripts

Validate dataset structure:
python data_utils.py --action validate --dataset-dir dataset
Analyze dataset statistics:
python data_utils.py --action analyze --dataset-dir dataset
Split dataset (if needed):
python data_utils.py --action split_dataset --images-dir raw_images --labels-dir raw_labels --output-dir dataset
Visualize annotations:
python data_utils.py --action visualize --images-dir dataset/train/images --labels-dir dataset/train/labels

🎯 Training

Basic Training

python train.py --config config.yaml --model-size n
Advanced Training Options

# Train with larger model
python train.py --config config.yaml --model-size m

# Resume training from checkpoint
python train.py --config config.yaml --model-size n --resume

# Validate dataset only
python train.py --config config.yaml --validate-only
Training Configuration

Edit config.yaml to customize:

Model size (n, s, m, l, x)
Training epochs and batch size
Learning rate and optimization
Data augmentation parameters
Validation settings
Training Outputs

Best model: runs/train/yolov8_training/weights/best.pt
Last model: runs/train/yolov8_training/weights/last.pt
Training curves: training_curves.png
Training summary: training_summary.txt
Logs: training.log
📊 Evaluation

Model Evaluation

python predict.py --model runs/train/yolov8_training/weights/best.pt --config config.yaml
Comprehensive Evaluation

# Evaluate on test set
python predict.py --model runs/train/yolov8_training/weights/best.pt --data config.yaml

# Generate predictions on images
python predict.py --model runs/train/yolov8_training/weights/best.pt --images dataset/test/images

# Analyze failure cases
python predict.py --model runs/train/yolov8_training/weights/best.pt --images dataset/test/images --analyze-failures
Evaluation Outputs

Confusion matrix: confusion_matrix.png
Per-class metrics: class_metrics.png
Evaluation summary: evaluation_summary.txt
Failure cases: failure_cases.json
Optimization recommendations: optimization_recommendations.txt
🌐 Web Application

Launch Streamlit App

streamlit run app.py
Features

Image Upload: Upload and detect objects in images
Webcam Detection: Real-time detection using webcam
Analytics Dashboard: Model performance and statistics
Interactive Settings: Adjust confidence threshold and model selection
Usage

Load a trained model from the sidebar
Adjust confidence threshold as needed
Upload images or use webcam for detection
View detection results and statistics
Analyze performance in the analytics tab
📁 Project Structure

space-station-object-detection/
├── config.yaml              # Main configuration file
├── environment.yml          # Conda environment
├── setup.py                # Setup script
├── train.py                # Training script
├── predict.py              # Evaluation script
├── app.py                  # Streamlit web app
├── data_utils.py           # Data utilities
├── README.md               # This file
├── dataset/                # Dataset directory
│   ├── train/
│   ├── val/
│   └── test/
├── runs/                   # Training outputs
│   ├── train/
│   └── val/
├── logs/                   # Log files
├── models/                 # Pre-trained models
└── results/                # Evaluation results
⚙️ Configuration

Key Configuration Parameters

Model Settings

model: yolov8n.pt          # Model size (n, s, m, l, x)
epochs: 100                # Training epochs
batch_size: 16             # Batch size
imgsz: 640                 # Input image size
Training Hyperparameters

lr0: 0.01                 # Initial learning rate
lrf: 0.01                 # Final learning rate
momentum: 0.937           # SGD momentum
weight_decay: 0.0005      # Weight decay
Data Augmentation

hsv_h: 0.015             # HSV-Hue augmentation
hsv_s: 0.7               # HSV-Saturation augmentation
hsv_v: 0.4               # HSV-Value augmentation
mosaic: 1.0              # Mosaic augmentation probability
fliplr: 0.5              # Horizontal flip probability
Classes

names:
  0: Toolbox
  1: Oxygen Tank
  2: Fire Extinguisher
🎯 Model Optimization Strategies

For Achieving ≥90% mAP@0.5

Data Quality

Diverse Lighting: Include various lighting conditions
Occlusion Scenarios: Add partial occlusions
Viewing Angles: Multiple camera angles
Class Balance: Ensure balanced class distribution
Training Strategies

Larger Models: Use YOLOv8s, YOLOv8m, or YOLOv8l
Extended Training: Increase epochs to 200-300
Learning Rate: Use cosine annealing
Data Augmentation: Increase mosaic and HSV augmentation
Advanced Techniques

Transfer Learning: Pre-train on COCO dataset
Ensemble Methods: Combine multiple models
Test Time Augmentation: Use TTA during inference
Model Pruning: Optimize for speed vs accuracy trade-off
🔧 Troubleshooting

Common Issues

CUDA Out of Memory

# Reduce batch size in config.yaml
batch_size: 8  # or smaller
Training Not Converging

# Check learning rate
lr0: 0.001  # Try smaller learning rate

# Increase epochs
epochs: 200
Poor Detection Performance

# Use larger model
model: yolov8s.pt  # or yolov8m.pt

# Increase data augmentation
mosaic: 1.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
Dataset Issues

# Validate dataset
python data_utils.py --action validate --dataset-dir dataset

# Check class balance
python data_utils.py --action analyze --dataset-dir dataset
Performance Monitoring

Training Metrics

Loss curves: Monitor training and validation loss
mAP@0.5: Target ≥90%
Per-class performance: Check individual class accuracy
System Resources

GPU memory: Monitor with nvidia-smi
CPU usage: Check with htop
Disk space: Ensure sufficient storage for logs
📈 Performance Benchmarks

Target Metrics

mAP@0.5: ≥90%
mAP@0.5:0.95: ≥70%
Precision: ≥85%
Recall: ≥85%
Expected Performance by Model Size

Model	mAP@0.5	Speed (ms)	Memory (GB)
YOLOv8n	85-90%	8.7	3.2
YOLOv8s	88-92%	12.9	11.2
YOLOv8m	90-94%	25.9	25.9
YOLOv8l	92-96%	43.7	43.7

🤝 Contributing

Fork the repository
Create a feature branch
Make your changes
Add tests if applicable
Submit a pull request

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

Ultralytics for YOLOv8 implementation
Duality AI for Falcon synthetic data platform
Streamlit for web application framework
OpenCV for computer vision utilities

📞 Support

For questions and support:

Create an issue on GitHub
Check the troubleshooting section
Review the documentation

Happy detecting! 🚀🔍
