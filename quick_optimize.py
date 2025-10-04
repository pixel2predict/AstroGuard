from ultralytics import YOLO
import os

if __name__ == '__main__': 
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    # Use YOLOv8m for better performance
    model = YOLO("yolov8m.pt")
    
    # Quick high-performance training
    results = model.train(
        data="yolo_params.yaml",
        epochs=20,  # Quick training
        device='cpu',
        imgsz=640,
        batch=4,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Enhanced augmentation for better performance
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.1,
        auto_augment='randaugment',
        erasing=0.4,
        # Quick settings
        patience=10,
        save_period=5,
        val=True,
        plots=True,
        save=True,
        verbose=True,
        name='quick_optimize'
    ) 