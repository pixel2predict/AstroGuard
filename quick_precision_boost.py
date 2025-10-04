"""
Quick Precision Boost Training Script
Uses current best model to quickly improve precision above 95%
"""

import os
import sys
from ultralytics import YOLO
import torch

def quick_precision_training():
    """Quick training to boost precision"""
    
    print("🚀 QUICK PRECISION BOOST TRAINING")
    print("=" * 50)
    
    # Check for best model
    best_model_path = "runs/detect/train/weights/best.pt"
    if not os.path.exists(best_model_path):
        print("❌ Best model not found. Please run regular training first.")
        return
    
    print(f"✅ Using best model: {best_model_path}")
    
    # Load model
    model = YOLO(best_model_path)
    
    print("🎯 Starting precision-focused fine-tuning...")
    
    # Quick precision training with conservative settings
    results = model.train(
        data="yolo_params.yaml",
        epochs=20,  # Quick training
        imgsz=1024,  # Larger image size for precision
        batch=4,     # Smaller batch for stability
        
        # Very conservative learning rate for precision
        lr0=0.0001,
        lrf=0.001,
        momentum=0.95,
        weight_decay=0.002,
        
        # Minimal augmentation for precision
        mosaic=0.0,      # Disable mosaic
        mixup=0.0,       # Disable mixup
        copy_paste=0.0,  # Disable copy-paste
        
        hsv_h=0.005,     # Minimal hue
        hsv_s=0.3,       # Minimal saturation
        hsv_v=0.2,       # Minimal value
        degrees=1.0,     # Minimal rotation
        translate=0.02,  # Minimal translation
        scale=0.95,      # Minimal scale
        shear=0.0,       # No shear
        perspective=0.0, # No perspective
        flipud=0.0,      # No vertical flip
        fliplr=0.5,      # Keep horizontal flip
        
        # Precision-focused loss weights
        box=10.0,        # High box loss for precise localization
        cls=0.3,         # Lower class loss
        dfl=1.5,         # Standard DFL loss
        
        # Training settings
        patience=15,
        save_period=5,
        optimizer='AdamW',
        cos_lr=True,
        
        # Project settings
        project='runs/train',
        name='precision_boost',
        resume=False,  # Don't resume, start fresh
        
        # Additional precision settings
        conf=0.25,       # Lower confidence threshold during training
        iou=0.5,         # Standard IoU threshold
        max_det=300      # Maximum detections
    )
    
    print(f"✅ Training completed!")
    print(f"📁 Results saved at: {results.save_dir}")
    print(f"🏆 Best model: {results.save_dir}/weights/best.pt")
    
    # Quick evaluation
    print("\n🔍 Quick evaluation...")
    eval_results = model.val(data="yolo_params.yaml", split='test')
    
    print(f"📊 Final Metrics:")
    print(f"   Precision: {eval_results.box.mp:.4f} ({eval_results.box.mp*100:.2f}%)")
    print(f"   Recall: {eval_results.box.mr:.4f} ({eval_results.box.mr*100:.2f}%)")
    print(f"   mAP@0.5: {eval_results.box.map50:.4f} ({eval_results.box.map50*100:.2f}%)")
    print(f"   mAP@0.5-0.95: {eval_results.box.map:.4f} ({eval_results.box.map*100:.2f}%)")
    
    # Precision assessment
    if eval_results.box.mp > 0.95:
        print("🎉 EXCELLENT! Precision > 95% achieved!")
    elif eval_results.box.mp > 0.90:
        print("✅ GOOD! Precision > 90% achieved!")
    else:
        print("⚠️ Precision needs improvement. Consider longer training.")

if __name__ == "__main__":
    quick_precision_training() 