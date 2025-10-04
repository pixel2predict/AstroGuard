"""
Improved YOLOv8 Training Script for Better Recall and mAP@0.5-0.95
Optimizations:
1. Extended training epochs
2. Better learning rate schedule
3. Enhanced data augmentation
4. Loss function tuning for recall
5. Model architecture upgrade option
"""

import argparse
from ultralytics import YOLO
import os
import sys

# Improved hyperparameters for better recall and mAP@0.5-0.95
EPOCHS = 50  # Increased from 20
MOSAIC = 0.7  # Slightly increased for better augmentation
OPTIMIZER = 'AdamW'
MOMENTUM = 0.937
LR0 = 0.005  # Reduced initial LR for more stable training
LRF = 0.1    # Higher final LR ratio for better convergence
SINGLE_CLS = False
MIXUP = 0.15  # Increased mixup for better generalization
COPY_PASTE = 0.3  # Increased copy-paste augmentation
CLOSE_MOSAIC = 10  # Disable mosaic in last 10 epochs for better precision

# Additional hyperparameters for recall improvement
HSV_H = 0.02  # Increased hue augmentation
HSV_S = 0.8   # Increased saturation augmentation
HSV_V = 0.5   # Increased value augmentation
DEGREES = 10.0  # Added rotation augmentation
TRANSLATE = 0.2  # Increased translation
SCALE = 0.9     # Increased scale augmentation
FLIPUD = 0.5    # Added vertical flip
FLIPLR = 0.5    # Added horizontal flip

# Loss function weights for better recall
BOX_GAIN = 7.5
CLS_GAIN = 0.3  # Reduced class loss to emphasize detection
DFL_GAIN = 1.5

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--model', type=str, default='yolov8l.pt', help='Model size (yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    parser.add_argument('--imgsz', type=int, default=832, help='Image size (increased for better detection)')
    parser.add_argument('--batch', type=int, default=8, help='Batch size (reduced due to larger image size)')
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    parser.add_argument('--mixup', type=float, default=MIXUP, help='Mixup augmentation')
    parser.add_argument('--copy_paste', type=float, default=COPY_PASTE, help='Copy paste augmentation')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
    
    args = parser.parse_args()
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    # Load model - option to use larger model for better performance
    model_path = os.path.join(this_dir, args.model)
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found, downloading...")
        model = YOLO(args.model)  # Will download if not exists
    else:
        model = YOLO(model_path)
    
    print(f"Training with model: {args.model}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Epochs: {args.epochs}")
    
    # Enhanced training with improved hyperparameters
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"), 
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        
        # Learning rate schedule
        optimizer=args.optimizer, 
        lr0=args.lr0, 
        lrf=args.lrf, 
        momentum=args.momentum,
        
        # Data augmentation for better recall
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        close_mosaic=CLOSE_MOSAIC,
        
        # Enhanced augmentations
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        
        # Loss function tuning for recall
        box=BOX_GAIN,
        cls=CLS_GAIN,
        dfl=DFL_GAIN,
        
        # Additional settings
        patience=15,  # Early stopping patience
        save_period=5,  # Save checkpoint every 5 epochs
        val=True,
        plots=True,
        verbose=True,
        
        # Resume from best checkpoint if exists
        resume=True,
        
        # Project name for organized results
        project='runs/train',
        name='improved_training'
    )
    
    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print(f"Results saved at: {results.save_dir}")

# Performance improvement notes:
"""
Expected improvements:
1. Recall: 85% → 90%+ (better augmentation + loss tuning)
2. mAP@0.5-0.95: 79% → 85%+ (larger model + extended training)
3. Overall robustness through enhanced augmentation

Key changes:
- Larger image size (832 vs 640) for better small object detection
- Extended training (50 epochs vs 20)
- Better learning rate schedule
- Enhanced data augmentation
- Loss function tuning to prioritize recall
- Option to use larger model (yolov8l.pt)
"""
