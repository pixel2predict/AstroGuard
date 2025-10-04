EPOCHS = 100
MOSAIC = 0.5
OPTIMIZER = 'AdamW'
MOMENTUM = 0.937
LR0 = 0.01
LRF = 0.01
SINGLE_CLS = False
import argparse
from ultralytics import YOLO
import os
import sys

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    args = parser.parse_args()
    
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    # Load pre-trained model
    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))
    
    # Optimized training parameters
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"), 
        epochs=args.epochs,
        device='cpu',
        single_cls=args.single_cls, 
        mosaic=args.mosaic,
        optimizer=args.optimizer, 
        lr0=args.lr0, 
        lrf=args.lrf, 
        momentum=args.momentum,
        # Enhanced augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mixup=0.1,
        copy_paste=0.1,
        # Better training settings
        batch=8,
        imgsz=640,
        patience=50,
        save_period=10,
        # Enhanced loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Advanced settings
        close_mosaic=10,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
        save=True,
        save_txt=True,
        save_conf=True,
        save_crop=False,
        verbose=True,
        seed=42,
        deterministic=True,
        workers=4
    ) 