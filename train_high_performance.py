import argparse
from ultralytics import YOLO
import os

if __name__ == '__main__': 
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    # Use YOLOv8m (medium) for better performance
    model = YOLO("yolov8m.pt")
    
    # High-performance training configuration
    results = model.train(
        data="yolo_params.yaml",
        epochs=50,  # More epochs for better convergence
        device='cpu',
        imgsz=640,
        batch=4,  # Smaller batch size for CPU
        optimizer='AdamW',
        lr0=0.001,  # Lower learning rate
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
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
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        cache=False,
        workers=4,
        project=None,
        name='train_high_performance',
        exist_ok=False,
        pretrained=True,
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
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.1,
        auto_augment='randaugment',
        erasing=0.4,
        crop_fraction=1.0
    ) 