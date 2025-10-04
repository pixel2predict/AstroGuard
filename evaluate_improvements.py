"""
Comprehensive Evaluation Script for YOLOv8 Model Performance
Tracks improvements in Recall and mAP@0.5-0.95 metrics
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import yaml
from pathlib import Path

class ModelEvaluator:
    def __init__(self, project_dir="."):
        self.project_dir = Path(project_dir)
        self.results_history = []
        
    def load_training_results(self, results_path):
        """Load training results from CSV file"""
        try:
            df = pd.read_csv(results_path)
            return df
        except Exception as e:
            print(f"Error loading results: {e}")
            return None
    
    def compare_models(self, baseline_path, improved_path):
        """Compare baseline vs improved model performance"""
        baseline_df = self.load_training_results(baseline_path)
        improved_df = self.load_training_results(improved_path)
        
        if baseline_df is None or improved_df is None:
            print("Could not load one or both result files")
            return
        
        # Get final epoch metrics
        baseline_final = baseline_df.iloc[-1]
        improved_final = improved_df.iloc[-1]
        
        print("=== MODEL PERFORMANCE COMPARISON ===")
        print(f"{'Metric':<20} {'Baseline':<12} {'Improved':<12} {'Change':<12}")
        print("-" * 60)
        
        metrics = [
            ('mAP@0.5', 'metrics/mAP50(B)'),
            ('mAP@0.5-0.95', 'metrics/mAP50-95(B)'),
            ('Precision', 'metrics/precision(B)'),
            ('Recall', 'metrics/recall(B)')
        ]
        
        for metric_name, metric_key in metrics:
            baseline_val = baseline_final[metric_key] * 100
            improved_val = improved_final[metric_key] * 100
            change = improved_val - baseline_val
            change_str = f"+{change:.2f}%" if change > 0 else f"{change:.2f}%"
            
            print(f"{metric_name:<20} {baseline_val:.2f}%{'':<6} {improved_val:.2f}%{'':<6} {change_str:<12}")
        
        return {
            'baseline': baseline_final,
            'improved': improved_final,
            'improvements': {
                'recall': (improved_final['metrics/recall(B)'] - baseline_final['metrics/recall(B)']) * 100,
                'map50_95': (improved_final['metrics/mAP50-95(B)'] - baseline_final['metrics/mAP50-95(B)']) * 100
            }
        }
    
    def plot_training_curves(self, results_paths, labels, save_path=None):
        """Plot training curves for comparison"""
        plt.figure(figsize=(15, 10))
        
        # Define subplots for different metrics
        metrics = [
            ('metrics/mAP50(B)', 'mAP@0.5'),
            ('metrics/mAP50-95(B)', 'mAP@0.5-0.95'),
            ('metrics/recall(B)', 'Recall'),
            ('metrics/precision(B)', 'Precision')
        ]
        
        for i, (metric_key, metric_name) in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            
            for results_path, label in zip(results_paths, labels):
                df = self.load_training_results(results_path)
                if df is not None:
                    plt.plot(df['epoch'], df[metric_key] * 100, label=label, linewidth=2)
            
            plt.title(f'{metric_name} Progress', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel(f'{metric_name} (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model_on_test_set(self, model_path, test_data_path):
        """Evaluate model on test set"""
        try:
            model = YOLO(model_path)
            results = model.val(data=test_data_path, split='test')
            
            print("=== TEST SET EVALUATION ===")
            print(f"mAP@0.5: {results.box.map50:.4f} ({results.box.map50*100:.2f}%)")
            print(f"mAP@0.5-0.95: {results.box.map:.4f} ({results.box.map*100:.2f}%)")
            print(f"Precision: {results.box.mp:.4f} ({results.box.mp*100:.2f}%)")
            print(f"Recall: {results.box.mr:.4f} ({results.box.mr*100:.2f}%)")
            
            # Per-class metrics
            print("\n=== PER-CLASS METRICS ===")
            class_names = ['FireExtinguisher', 'ToolBox', 'OxygenTank']
            for i, class_name in enumerate(class_names):
                if i < len(results.box.ap50):
                    print(f"{class_name}: mAP@0.5 = {results.box.ap50[i]:.4f} ({results.box.ap50[i]*100:.2f}%)")
            
            return results
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return None
    
    def generate_improvement_report(self, baseline_results, improved_results):
        """Generate detailed improvement report"""
        report = f"""
# YOLOv8 Model Improvement Report

## Objective
Improve Recall from 85.07% to 90%+ and mAP@0.5-0.95 from 79.38% to 85%+

## Implemented Improvements

### 1. Model Architecture
- Upgraded from YOLOv8m to YOLOv8l for better feature extraction
- Increased image size from 640 to 832 pixels

### 2. Training Optimization
- Extended training from 20 to 50 epochs
- Improved learning rate schedule (lr0: 0.01→0.005, lrf: 0.01→0.1)
- Added cosine learning rate scheduler

### 3. Data Augmentation Enhancement
- Increased mosaic probability: 0.5 → 0.7
- Enhanced HSV augmentation
- Added rotation, shear, and flip augmentations
- Increased mixup and copy-paste probabilities

### 4. Loss Function Tuning
- Reduced classification loss weight (0.5 → 0.3) to emphasize detection
- Added label smoothing (0.1) for better generalization

## Expected Results
- **Recall**: 85.07% → 90%+ (Target: +5% improvement)
- **mAP@0.5-0.95**: 79.38% → 85%+ (Target: +6% improvement)
- **Maintained high precision**: ~96%

## Next Steps
1. Run improved training script: `python train_improved.py`
2. Monitor training progress in runs/train/improved_training
3. Evaluate on test set using this script
4. Fine-tune hyperparameters if needed

## Usage Instructions
```bash
# Start improved training
python train_improved.py --epochs 50 --model yolov8l.pt

# Evaluate improvements
python evaluate_improvements.py
```
        """
        
        with open(self.project_dir / "improvement_report.md", "w") as f:
            f.write(report)
        
        print("Improvement report saved to improvement_report.md")

def main():
    evaluator = ModelEvaluator()
    
    # Check for existing results
    baseline_results = "runs/detect/train/results.csv"
    improved_results = "runs/train/improved_training/results.csv"
    
    if os.path.exists(baseline_results):
        print("Found baseline training results!")
        
        if os.path.exists(improved_results):
            print("Found improved training results!")
            # Compare models
            comparison = evaluator.compare_models(baseline_results, improved_results)
            
            # Plot training curves
            evaluator.plot_training_curves(
                [baseline_results, improved_results],
                ['Baseline (YOLOv8m)', 'Improved (YOLOv8l)'],
                'training_comparison.png'
            )
            
        else:
            print("Improved training results not found. Run train_improved.py first.")
    else:
        print("Baseline results not found.")
    
    # Generate improvement report
    evaluator.generate_improvement_report(None, None)
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. Run: python train_improved.py --epochs 50 --model yolov8l.pt")
    print("2. Monitor training in runs/train/improved_training/")
    print("3. Compare results using this evaluation script")
    print("4. Test on validation set for final metrics")

if __name__ == "__main__":
    main()
