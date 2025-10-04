"""
Precision-Focused Model Evaluation Script
Analyzes model performance with emphasis on precision metrics
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import numpy as np
from pathlib import Path

class PrecisionEvaluator:
    def __init__(self, project_dir="."):
        self.project_dir = Path(project_dir)
        
    def analyze_training_results(self, results_path):
        """Analyze training results with focus on precision"""
        if not os.path.exists(results_path):
            print(f"❌ Results file not found: {results_path}")
            return None
            
        df = pd.read_csv(results_path)
        
        print(f"📊 TRAINING RESULTS ANALYSIS: {results_path}")
        print("=" * 60)
        
        # Find best epoch by precision
        best_precision_epoch = df.loc[df['metrics/precision(B)'].idxmax()]
        best_map_epoch = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
        best_recall_epoch = df.loc[df['metrics/recall(B)'].idxmax()]
        final_epoch = df.iloc[-1]
        
        print(f"🏆 BEST PRECISION EPOCH ({best_precision_epoch['epoch']:.0f}):")
        print(f"   Precision: {best_precision_epoch['metrics/precision(B)']:.4f} ({best_precision_epoch['metrics/precision(B)']*100:.2f}%)")
        print(f"   Recall: {best_precision_epoch['metrics/recall(B)']:.4f} ({best_precision_epoch['metrics/recall(B)']*100:.2f}%)")
        print(f"   mAP@0.5: {best_precision_epoch['metrics/mAP50(B)']:.4f} ({best_precision_epoch['metrics/mAP50(B)']*100:.2f}%)")
        print(f"   mAP@0.5-0.95: {best_precision_epoch['metrics/mAP50-95(B)']:.4f} ({best_precision_epoch['metrics/mAP50-95(B)']*100:.2f}%)")
        
        print(f"\n🎯 BEST mAP EPOCH ({best_map_epoch['epoch']:.0f}):")
        print(f"   Precision: {best_map_epoch['metrics/precision(B)']:.4f} ({best_map_epoch['metrics/precision(B)']*100:.2f}%)")
        print(f"   Recall: {best_map_epoch['metrics/recall(B)']:.4f} ({best_map_epoch['metrics/recall(B)']*100:.2f}%)")
        print(f"   mAP@0.5: {best_map_epoch['metrics/mAP50(B)']:.4f} ({best_map_epoch['metrics/mAP50(B)']*100:.2f}%)")
        print(f"   mAP@0.5-0.95: {best_map_epoch['metrics/mAP50-95(B)']:.4f} ({best_map_epoch['metrics/mAP50-95(B)']*100:.2f}%)")
        
        print(f"\n📈 FINAL EPOCH ({final_epoch['epoch']:.0f}):")
        print(f"   Precision: {final_epoch['metrics/precision(B)']:.4f} ({final_epoch['metrics/precision(B)']*100:.2f}%)")
        print(f"   Recall: {final_epoch['metrics/recall(B)']:.4f} ({final_epoch['metrics/recall(B)']*100:.2f}%)")
        print(f"   mAP@0.5: {final_epoch['metrics/mAP50(B)']:.4f} ({final_epoch['metrics/mAP50(B)']*100:.2f}%)")
        print(f"   mAP@0.5-0.95: {final_epoch['metrics/mAP50-95(B)']:.4f} ({final_epoch['metrics/mAP50-95(B)']*100:.2f}%)")
        
        # Precision analysis
        precision_above_90 = df[df['metrics/precision(B)'] > 0.9]
        precision_above_95 = df[df['metrics/precision(B)'] > 0.95]
        
        print(f"\n🎯 PRECISION ANALYSIS:")
        print(f"   Epochs with Precision > 90%: {len(precision_above_90)}/{len(df)}")
        print(f"   Epochs with Precision > 95%: {len(precision_above_95)}/{len(df)}")
        print(f"   Average Precision: {df['metrics/precision(B)'].mean():.4f} ({df['metrics/precision(B)'].mean()*100:.2f}%)")
        print(f"   Max Precision: {df['metrics/precision(B)'].max():.4f} ({df['metrics/precision(B)'].max()*100:.2f}%)")
        
        return {
            'best_precision': best_precision_epoch,
            'best_map': best_map_epoch,
            'final': final_epoch,
            'precision_above_90_count': len(precision_above_90),
            'precision_above_95_count': len(precision_above_95),
            'avg_precision': df['metrics/precision(B)'].mean(),
            'max_precision': df['metrics/precision(B)'].max()
        }
    
    def plot_precision_analysis(self, results_paths, labels):
        """Plot precision-focused analysis"""
        plt.figure(figsize=(15, 12))
        
        for i, (results_path, label) in enumerate(zip(results_paths, labels)):
            if os.path.exists(results_path):
                df = pd.read_csv(results_path)
                
                # Precision over epochs
                plt.subplot(2, 2, 1)
                plt.plot(df['epoch'], df['metrics/precision(B)'] * 100, 
                        label=label, linewidth=2, marker='o', markersize=3)
                plt.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% Threshold')
                plt.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% Threshold')
                plt.title('Precision Progress', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch')
                plt.ylabel('Precision (%)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Recall vs Precision
                plt.subplot(2, 2, 2)
                plt.scatter(df['metrics/recall(B)'] * 100, df['metrics/precision(B)'] * 100, 
                           alpha=0.6, label=label, s=30)
                plt.xlabel('Recall (%)')
                plt.ylabel('Precision (%)')
                plt.title('Precision vs Recall', fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # mAP progress
                plt.subplot(2, 2, 3)
                plt.plot(df['epoch'], df['metrics/mAP50(B)'] * 100, 
                        label=f'{label} (mAP@0.5)', linewidth=2)
                plt.plot(df['epoch'], df['metrics/mAP50-95(B)'] * 100, 
                        label=f'{label} (mAP@0.5-0.95)', linewidth=2)
                plt.title('mAP Progress', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch')
                plt.ylabel('mAP (%)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Loss components
                plt.subplot(2, 2, 4)
                plt.plot(df['epoch'], df['train/box_loss'], label=f'{label} Box Loss', alpha=0.7)
                plt.plot(df['epoch'], df['train/cls_loss'], label=f'{label} Class Loss', alpha=0.7)
                plt.title('Training Loss Components', fontsize=14, fontweight='bold')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('precision_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model_on_test(self, model_path, test_data_path):
        """Evaluate model on test set with precision focus"""
        print(f"\n🔍 TEST SET EVALUATION: {model_path}")
        print("=" * 60)
        
        try:
            model = YOLO(model_path)
            results = model.val(data=test_data_path, split='test')
            
            print(f"📊 OVERALL METRICS:")
            print(f"   Precision: {results.box.mp:.4f} ({results.box.mp*100:.2f}%)")
            print(f"   Recall: {results.box.mr:.4f} ({results.box.mr*100:.2f}%)")
            print(f"   mAP@0.5: {results.box.map50:.4f} ({results.box.map50*100:.2f}%)")
            print(f"   mAP@0.5-0.95: {results.box.map:.4f} ({results.box.map*100:.2f}%)")
            
            # Per-class metrics
            print(f"\n🎯 PER-CLASS METRICS:")
            class_names = ['FireExtinguisher', 'ToolBox', 'OxygenTank']
            for i, class_name in enumerate(class_names):
                if i < len(results.box.ap50):
                    print(f"   {class_name}:")
                    print(f"     mAP@0.5: {results.box.ap50[i]:.4f} ({results.box.ap50[i]*100:.2f}%)")
                    if i < len(results.box.ap):
                        print(f"     mAP@0.5-0.95: {results.box.ap[i]:.4f} ({results.box.ap[i]*100:.2f}%)")
            
            # Precision assessment
            precision_status = "✅ EXCELLENT" if results.box.mp > 0.95 else \
                             "✅ GOOD" if results.box.mp > 0.90 else \
                             "⚠️ NEEDS IMPROVEMENT" if results.box.mp > 0.80 else "❌ POOR"
            
            print(f"\n🎯 PRECISION ASSESSMENT: {precision_status}")
            print(f"   Target: > 90% | Current: {results.box.mp*100:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"❌ Error evaluating model: {e}")
            return None
    
    def generate_precision_report(self, results_analysis, test_results):
        """Generate precision-focused report"""
        report = f"""
# High Precision Model Evaluation Report

## Summary
- **Model Performance**: {'Excellent' if test_results and test_results.box.mp > 0.95 else 'Good' if test_results and test_results.box.mp > 0.90 else 'Needs Improvement'}
- **Precision Target**: > 90%
- **Current Precision**: {test_results.box.mp*100:.2f}% if test_results else 'N/A'

## Training Analysis
- **Best Precision**: {results_analysis['max_precision']*100:.2f}%
- **Average Precision**: {results_analysis['avg_precision']*100:.2f}%
- **Epochs with Precision > 90%**: {results_analysis['precision_above_90_count']}
- **Epochs with Precision > 95%**: {results_analysis['precision_above_95_count']}

## Recommendations
1. {'✅ Model meets precision requirements' if test_results and test_results.box.mp > 0.90 else '⚠️ Consider additional training or hyperparameter tuning'}
2. {'✅ Excellent performance achieved' if test_results and test_results.box.mp > 0.95 else '🎯 Room for improvement'}
3. Monitor precision during deployment

## Next Steps
- Deploy model if precision requirements are met
- Consider ensemble methods for further improvement
- Regular retraining with new data
        """
        
        with open(self.project_dir / "precision_report.md", "w", encoding='utf-8') as f:
            f.write(report)
        
        print("📄 Precision report saved to precision_report.md")

def main():
    evaluator = PrecisionEvaluator()
    
    # Analyze existing training results
    training_results = [
        "runs/detect/train/results.csv",
        "runs/detect/train2/results.csv"
    ]
    
    results_labels = ["Training Run 1", "Training Run 2"]
    
    print("=== PRECISION-FOCUSED MODEL EVALUATION ===\n")
    
    # Analyze each training run
    analyses = []
    for results_path, label in zip(training_results, results_labels):
        if os.path.exists(results_path):
            print(f"Analyzing {label}...")
            analysis = evaluator.analyze_training_results(results_path)
            if analysis:
                analyses.append(analysis)
            print("\n" + "="*60 + "\n")
    
    # Plot analysis
    if len(analyses) > 0:
        evaluator.plot_precision_analysis(training_results, results_labels)
    
    # Evaluate best model on test set
    best_model_path = "runs/detect/train/weights/best.pt"
    if os.path.exists(best_model_path):
        test_results = evaluator.evaluate_model_on_test(best_model_path, "yolo_params.yaml")
        
        # Generate report
        if analyses:
            evaluator.generate_precision_report(analyses[0], test_results)
    else:
        print("❌ Best model not found. Please run training first.")
    
    print("\n💡 RECOMMENDATIONS:")
    print("1. Use train_high_precision.py for >90% precision training")
    print("2. Monitor precision during training")
    print("3. Consider ensemble methods for production deployment")

if __name__ == "__main__":
    main() 