"""
High Precision YOLOv8 Training Script
Optimized for >90% Precision while maintaining good recall
Features:
1. Extended training with precision-focused hyperparameters
2. Conservative data augmentation
3. Advanced regularization techniques
4. Multi-stage training approach
5. Precision-focused loss function tuning
"""

import argparse
import os
import sys
import yaml
from ultralytics import YOLO
import torch
import numpy as np
from pathlib import Path

class HighPrecisionTrainer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
    def create_precision_config(self, epochs=100, model_size='yolov8x.pt'):
        """Create precision-optimized configuration"""
        config = {
            'path': './',
            'train': 'data/train/images',
            'val': 'data/val/images',
            'test': 'data/test/images',
            'nc': 3,
            'names': ['FireExtinguisher', 'ToolBox', 'OxygenTank'],
            
            # Model settings
            'model': model_size,
            'epochs': epochs,
            'batch_size': 4,
            'imgsz': 1024,
            'device': 'auto',
            
            # Precision-optimized hyperparameters
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.95,
            'weight_decay': 0.001,
            'warmup_epochs': 5.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Loss function tuning
            'box': 8.0,
            'cls': 0.5,
            'dfl': 1.5,
            'label_smoothing': 0.05,
            'nbs': 64,
            'dropout': 0.2,
            
            # Conservative augmentation
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 5.0,
            'translate': 0.1,
            'scale': 0.8,
            'shear': 1.0,
            'perspective': 0.0001,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.5,
            'mixup': 0.1,
            'copy_paste': 0.1,
            'close_mosaic': 20,
            
            # Training settings
            'patience': 25,
            'save_period': 10,
            'val': True,
            'plots': True,
            'verbose': True,
            
            # Advanced settings
            'multiscale': True,
            'scale_range': [0.9, 1.1],
            'tta': True,
            'optimizer': 'AdamW',
            'cos_lr': True,
            'erasing': 0.2,
            'crop_fraction': 1.0,
            'ema': True,
            'ema_decay': 0.9999,
            'auto_augment': 'randaugment',
            'augment': True,
            'conf': 0.25,
            'iou': 0.5,
            'max_det': 300
        }
        
        # Save configuration
        config_path = 'config_high_precision.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_path
    
    def train_stage1(self, model_path, config_path, epochs=50):
        """Stage 1: Foundation training with conservative settings"""
        print("=== STAGE 1: Foundation Training ===")
        
        model = YOLO(model_path)
        
        results = model.train(
            data=config_path,
            epochs=epochs,
            imgsz=1024,
            batch=4,
            device=self.device,
            
            # Conservative learning rate
            lr0=0.001,
            lrf=0.01,
            momentum=0.95,
            weight_decay=0.001,
            
            # Minimal augmentation for precision
            mosaic=0.3,
            mixup=0.05,
            copy_paste=0.05,
            close_mosaic=30,
            
            hsv_h=0.01,
            hsv_s=0.5,
            hsv_v=0.3,
            degrees=2.0,
            translate=0.05,
            scale=0.9,
            shear=0.5,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            
            # Precision-focused loss
            box=8.0,
            cls=0.5,
            dfl=1.5,
            
            # Training settings
            patience=20,
            save_period=10,
            optimizer='AdamW',
            cos_lr=True,
            
            project='runs/train',
            name='high_precision_stage1',
            resume=True
        )
        
        return results
    
    def train_stage2(self, best_model_path, config_path, epochs=50):
        """Stage 2: Fine-tuning with precision focus"""
        print("=== STAGE 2: Precision Fine-tuning ===")
        
        model = YOLO(best_model_path)
        
        results = model.train(
            data=config_path,
            epochs=epochs,
            imgsz=1024,
            batch=2,  # Smaller batch for fine-tuning
            device=self.device,
            
            # Very low learning rate for fine-tuning
            lr0=0.0001,
            lrf=0.001,
            momentum=0.95,
            weight_decay=0.002,
            
            # Minimal augmentation
            mosaic=0.0,  # Disable mosaic
            mixup=0.0,   # Disable mixup
            copy_paste=0.0,
            
            hsv_h=0.005,
            hsv_s=0.3,
            hsv_v=0.2,
            degrees=1.0,
            translate=0.02,
            scale=0.95,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            
            # Precision-focused loss
            box=10.0,
            cls=0.3,
            dfl=1.5,
            
            # Training settings
            patience=30,
            save_period=5,
            optimizer='AdamW',
            cos_lr=True,
            
            project='runs/train',
            name='high_precision_stage2',
            resume=True
        )
        
        return results
    
    def evaluate_model(self, model_path, test_data_path):
        """Evaluate model performance"""
        print("=== MODEL EVALUATION ===")
        
        model = YOLO(model_path)
        results = model.val(data=test_data_path, split='test')
        
        print(f"Precision: {results.box.mp:.4f} ({results.box.mp*100:.2f}%)")
        print(f"Recall: {results.box.mr:.4f} ({results.box.mr*100:.2f}%)")
        print(f"mAP@0.5: {results.box.map50:.4f} ({results.box.map50*100:.2f}%)")
        print(f"mAP@0.5-0.95: {results.box.map:.4f} ({results.box.map*100:.2f}%)")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='High Precision YOLOv8 Training')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--model', type=str, default='yolov8x.pt', help='Model size')
    parser.add_argument('--stage1_epochs', type=int, default=50, help='Stage 1 epochs')
    parser.add_argument('--stage2_epochs', type=int, default=50, help='Stage 2 epochs')
    parser.add_argument('--skip_stage1', action='store_true', help='Skip stage 1 training')
    
    args = parser.parse_args()
    
    trainer = HighPrecisionTrainer()
    
    # Create precision-optimized configuration
    config_path = trainer.create_precision_config(args.epochs, args.model)
    
    # Stage 1: Foundation training
    if not args.skip_stage1:
        stage1_results = trainer.train_stage1(args.model, config_path, args.stage1_epochs)
        best_model_path = stage1_results.save_dir / 'weights' / 'best.pt'
    else:
        best_model_path = args.model
    
    # Stage 2: Precision fine-tuning
    stage2_results = trainer.train_stage2(best_model_path, config_path, args.stage2_epochs)
    final_model_path = stage2_results.save_dir / 'weights' / 'best.pt'
    
    # Evaluate final model
    trainer.evaluate_model(final_model_path, 'yolo_params.yaml')
    
    print(f"\n=== TRAINING COMPLETED ===")
    print(f"Final model saved at: {final_model_path}")
    print(f"Results saved at: {stage2_results.save_dir}")

if __name__ == '__main__':
    main() 