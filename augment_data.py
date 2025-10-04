import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import random

def augment_image(image, bboxes, labels):
    """Apply various augmentations to image and adjust bounding boxes"""
    h, w = image.shape[:2]
    augmented_images = []
    augmented_bboxes = []
    augmented_labels = []
    
    # Original image
    augmented_images.append(image)
    augmented_bboxes.append(bboxes)
    augmented_labels.append(labels)
    
    # Horizontal flip
    flipped_img = cv2.flip(image, 1)
    flipped_bboxes = []
    for bbox in bboxes:
        x, y, width, height = bbox
        new_x = 1.0 - x - width  # Flip x coordinate
        flipped_bboxes.append([new_x, y, width, height])
    augmented_images.append(flipped_img)
    augmented_bboxes.append(flipped_bboxes)
    augmented_labels.append(labels)
    
    # Brightness adjustment
    bright_img = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    augmented_images.append(bright_img)
    augmented_bboxes.append(bboxes)
    augmented_labels.append(labels)
    
    # Contrast adjustment
    contrast_img = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
    augmented_images.append(contrast_img)
    augmented_bboxes.append(bboxes)
    augmented_labels.append(labels)
    
    # Slight rotation
    angle = random.uniform(-15, 15)
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h))
    # Note: Bounding box adjustment for rotation is complex, using original for simplicity
    augmented_images.append(rotated_img)
    augmented_bboxes.append(bboxes)
    augmented_labels.append(labels)
    
    return augmented_images, augmented_bboxes, augmented_labels

def parse_yolo_label(label_path):
    """Parse YOLO format label file"""
    bboxes = []
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                bboxes.append([x, y, w, h])
                labels.append(class_id)
    return bboxes, labels

def save_yolo_label(bboxes, labels, output_path):
    """Save bounding boxes in YOLO format"""
    with open(output_path, 'w') as f:
        for bbox, label in zip(bboxes, labels):
            x, y, w, h = bbox
            f.write(f"{label} {x} {y} {w} {h}\n")

def augment_dataset():
    """Augment the training dataset"""
    base_dir = Path("data/train")
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    
    # Create augmented dataset directory
    aug_dir = Path("data/train_augmented")
    aug_images_dir = aug_dir / "images"
    aug_labels_dir = aug_dir / "labels"
    
    aug_images_dir.mkdir(parents=True, exist_ok=True)
    aug_labels_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(images_dir.glob("*.png"))
    print(f"Found {len(image_files)} images to augment")
    
    total_augmented = 0
    
    for img_path in image_files:
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        # Load corresponding label
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
            
        bboxes, labels = parse_yolo_label(label_path)
        
        # Apply augmentations
        aug_images, aug_bboxes, aug_labels = augment_image(image, bboxes, labels)
        
        # Save augmented images and labels
        for i, (aug_img, aug_bbox, aug_label) in enumerate(zip(aug_images, aug_bboxes, aug_labels)):
            # Save image
            aug_img_path = aug_images_dir / f"{img_path.stem}_aug{i}.png"
            cv2.imwrite(str(aug_img_path), aug_img)
            
            # Save label
            aug_label_path = aug_labels_dir / f"{img_path.stem}_aug{i}.txt"
            save_yolo_label(aug_bbox, aug_label, aug_label_path)
            
            total_augmented += 1
    
    print(f"Augmentation complete! Generated {total_augmented} additional images")
    return aug_dir

if __name__ == "__main__":
    print("Starting dataset augmentation...")
    augmented_dir = augment_dataset()
    print(f"Augmented dataset saved to: {augmented_dir}") 