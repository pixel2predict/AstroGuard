#!/usr/bin/env python3
"""
Data Utilities for YOLOv8 Space Station Object Detection
Handles data preparation, validation, and augmentation
"""

import os
import yaml
import logging
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import random
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config_path='config.yaml'):
        """Initialize data processor with configuration."""
        self.config_path = config_path
        self.config = self.load_config()
        self.class_names = list(self.config['names'].values())
        
    def load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def validate_yolo_labels(self, label_path):
        """Validate YOLO format labels."""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    logger.warning(f"Invalid label format at line {line_num}: {line}")
                    return False
                
                class_id = int(parts[0])
                if class_id >= len(self.class_names):
                    logger.warning(f"Invalid class ID {class_id} at line {line_num}")
                    return False
                
                # Check normalized coordinates
                for i, coord in enumerate(parts[1:], 1):
                    try:
                        coord_val = float(coord)
                        if coord_val < 0 or coord_val > 1:
                            logger.warning(f"Coordinate {coord_val} out of range [0,1] at line {line_num}")
                            return False
                    except ValueError:
                        logger.warning(f"Invalid coordinate {coord} at line {line_num}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating labels: {e}")
            return False
    
    def visualize_annotations(self, image_path, label_path, save_path=None):
        """Visualize bounding boxes on image."""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            height, width = image.shape[:2]
            
            # Load labels
            if not label_path.exists():
                logger.warning(f"Label file not found: {label_path}")
                return False
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Draw bounding boxes
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:])
                    
                    # Convert normalized coordinates to pixel coordinates
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height
                    
                    # Calculate bounding box coordinates
                    x1 = int(x_center - w/2)
                    y1 = int(y_center - h/2)
                    x2 = int(x_center + w/2)
                    y2 = int(y_center + h/2)
                    
                    # Draw rectangle
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    class_name = self.class_names[class_id]
                    label = f"{class_name}"
                    cv2.putText(image, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save or display
            if save_path:
                cv2.imwrite(str(save_path), image)
                logger.info(f"Visualization saved to {save_path}")
            else:
                cv2.imshow('Annotations', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing annotations: {e}")
            return False
    
    def create_dataset_structure(self, output_dir='dataset'):
        """Create YOLO dataset directory structure."""
        try:
            dataset_path = Path(output_dir)
            
            # Create directories
            splits = ['train', 'val', 'test']
            for split in splits:
                (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
                (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Dataset structure created at {dataset_path}")
            return dataset_path
            
        except Exception as e:
            logger.error(f"Error creating dataset structure: {e}")
            return None
    
    def split_dataset(self, images_dir, labels_dir, output_dir='dataset', 
                     train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Split dataset into train/val/test sets."""
        try:
            # Get all image files
            images_path = Path(images_dir)
            image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
            
            if not image_files:
                logger.error("No image files found!")
                return False
            
            # Get corresponding label files
            labels_path = Path(labels_dir)
            valid_pairs = []
            
            for img_file in image_files:
                label_file = labels_path / f"{img_file.stem}.txt"
                if label_file.exists():
                    valid_pairs.append((img_file, label_file))
                else:
                    logger.warning(f"No label file for {img_file}")
            
            if not valid_pairs:
                logger.error("No valid image-label pairs found!")
                return False
            
            # Split dataset
            random.shuffle(valid_pairs)
            
            n_total = len(valid_pairs)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_pairs = valid_pairs[:n_train]
            val_pairs = valid_pairs[n_train:n_train + n_val]
            test_pairs = valid_pairs[n_train + n_val:]
            
            # Create output directory
            dataset_path = self.create_dataset_structure(output_dir)
            if dataset_path is None:
                return False
            
            # Copy files to appropriate directories
            splits = [
                ('train', train_pairs),
                ('val', val_pairs),
                ('test', test_pairs)
            ]
            
            for split_name, pairs in splits:
                for img_file, label_file in pairs:
                    # Copy image
                    dst_img = dataset_path / split_name / 'images' / img_file.name
                    shutil.copy2(img_file, dst_img)
                    
                    # Copy label
                    dst_label = dataset_path / split_name / 'labels' / label_file.name
                    shutil.copy2(label_file, dst_label)
                
                logger.info(f"{split_name}: {len(pairs)} pairs")
            
            return True
            
        except Exception as e:
            logger.error(f"Error splitting dataset: {e}")
            return False
    
    def analyze_dataset(self, dataset_dir='dataset'):
        """Analyze dataset statistics."""
        try:
            dataset_path = Path(dataset_dir)
            
            if not dataset_path.exists():
                logger.error(f"Dataset directory not found: {dataset_path}")
                return False
            
            stats = {}
            
            for split in ['train', 'val', 'test']:
                images_dir = dataset_path / split / 'images'
                labels_dir = dataset_path / split / 'labels'
                
                if not images_dir.exists() or not labels_dir.exists():
                    logger.warning(f"Split {split} not found")
                    continue
                
                # Count files
                image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
                label_files = list(labels_dir.glob('*.txt'))
                
                # Count annotations per class
                class_counts = {i: 0 for i in range(len(self.class_names))}
                total_annotations = 0
                
                for label_file in label_files:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            if class_id < len(self.class_names):
                                class_counts[class_id] += 1
                                total_annotations += 1
                
                stats[split] = {
                    'images': len(image_files),
                    'labels': len(label_files),
                    'annotations': total_annotations,
                    'class_counts': class_counts
                }
            
            # Print statistics
            logger.info("Dataset Statistics:")
            logger.info("=" * 50)
            
            for split, data in stats.items():
                logger.info(f"\n{split.upper()}:")
                logger.info(f"  Images: {data['images']}")
                logger.info(f"  Labels: {data['labels']}")
                logger.info(f"  Total Annotations: {data['annotations']}")
                logger.info("  Per-class annotations:")
                for class_id, count in data['class_counts'].items():
                    class_name = self.class_names[class_id]
                    logger.info(f"    {class_name}: {count}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing dataset: {e}")
            return None
    
    def plot_dataset_distribution(self, stats, save_path='dataset_distribution.png'):
        """Plot dataset distribution charts."""
        try:
            if not stats:
                logger.warning("No statistics available for plotting")
                return False
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Dataset Distribution Analysis', fontsize=16)
            
            # Split distribution
            splits = list(stats.keys())
            image_counts = [stats[split]['images'] for split in splits]
            axes[0, 0].bar(splits, image_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[0, 0].set_title('Images per Split')
            axes[0, 0].set_ylabel('Number of Images')
            
            # Annotation distribution
            annotation_counts = [stats[split]['annotations'] for split in splits]
            axes[0, 1].bar(splits, annotation_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[0, 1].set_title('Annotations per Split')
            axes[0, 1].set_ylabel('Number of Annotations')
            
            # Class distribution (train set)
            if 'train' in stats:
                train_stats = stats['train']
                class_names = self.class_names
                class_counts = [train_stats['class_counts'][i] for i in range(len(class_names))]
                
                axes[1, 0].bar(class_names, class_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                axes[1, 0].set_title('Class Distribution (Train Set)')
                axes[1, 0].set_ylabel('Number of Annotations')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Average annotations per image
            avg_annotations = []
            for split in splits:
                if stats[split]['images'] > 0:
                    avg = stats[split]['annotations'] / stats[split]['images']
                    avg_annotations.append(avg)
                else:
                    avg_annotations.append(0)
            
            axes[1, 1].bar(splits, avg_annotations, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[1, 1].set_title('Average Annotations per Image')
            axes[1, 1].set_ylabel('Average Annotations')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dataset distribution plot saved to {save_path}")
            plt.show()
            
            return True
            
        except Exception as e:
            logger.error(f"Error plotting dataset distribution: {e}")
            return False
    
    def validate_dataset_integrity(self, dataset_dir='dataset'):
        """Validate dataset integrity and consistency."""
        try:
            dataset_path = Path(dataset_dir)
            
            if not dataset_path.exists():
                logger.error(f"Dataset directory not found: {dataset_path}")
                return False
            
            issues = []
            
            for split in ['train', 'val', 'test']:
                images_dir = dataset_path / split / 'images'
                labels_dir = dataset_path / split / 'labels'
                
                if not images_dir.exists() or not labels_dir.exists():
                    issues.append(f"Missing directories for {split}")
                    continue
                
                # Get all files
                image_files = set(f.stem for f in images_dir.glob('*.jpg') + images_dir.glob('*.png'))
                label_files = set(f.stem for f in labels_dir.glob('*.txt'))
                
                # Check for missing labels
                missing_labels = image_files - label_files
                if missing_labels:
                    issues.append(f"{split}: {len(missing_labels)} images without labels")
                
                # Check for missing images
                missing_images = label_files - image_files
                if missing_images:
                    issues.append(f"{split}: {len(missing_images)} labels without images")
                
                # Validate label format
                for label_file in labels_dir.glob('*.txt'):
                    if not self.validate_yolo_labels(label_file):
                        issues.append(f"{split}: Invalid label format in {label_file.name}")
            
            if issues:
                logger.warning("Dataset integrity issues found:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
                return False
            else:
                logger.info("Dataset integrity validation passed!")
                return True
                
        except Exception as e:
            logger.error(f"Error validating dataset integrity: {e}")
            return False

def main():
    """Main function for data processing utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Processing Utilities for YOLOv8')
    parser.add_argument('--action', type=str, required=True, 
                       choices=['create_structure', 'split_dataset', 'analyze', 'validate', 'visualize'],
                       help='Action to perform')
    parser.add_argument('--images-dir', type=str, help='Directory containing images')
    parser.add_argument('--labels-dir', type=str, help='Directory containing labels')
    parser.add_argument('--dataset-dir', type=str, default='dataset', help='Dataset directory')
    parser.add_argument('--output-dir', type=str, default='dataset', help='Output directory')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DataProcessor(args.config)
    
    if args.action == 'create_structure':
        processor.create_dataset_structure(args.output_dir)
    
    elif args.action == 'split_dataset':
        if not args.images_dir or not args.labels_dir:
            logger.error("--images-dir and --labels-dir required for split_dataset")
            return
        processor.split_dataset(args.images_dir, args.labels_dir, args.output_dir)
    
    elif args.action == 'analyze':
        stats = processor.analyze_dataset(args.dataset_dir)
        if stats:
            processor.plot_dataset_distribution(stats)
    
    elif args.action == 'validate':
        processor.validate_dataset_integrity(args.dataset_dir)
    
    elif args.action == 'visualize':
        if not args.images_dir or not args.labels_dir:
            logger.error("--images-dir and --labels-dir required for visualize")
            return
        
        # Visualize first few images
        images_dir = Path(args.images_dir)
        labels_dir = Path(args.labels_dir)
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        for i, img_file in enumerate(image_files[:5]):  # First 5 images
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                save_path = f"visualization_{i}.png"
                processor.visualize_annotations(img_file, label_file, save_path)

if __name__ == "__main__":
    main() 