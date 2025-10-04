"""
Comprehensive Performance Analysis Script
Generates confusion matrix, performance graphs, and detailed analytics
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import yaml
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalyzer:
    def __init__(self, project_dir="."):
        self.project_dir = Path(project_dir)
        self.class_names = ['FireExtinguisher', 'ToolBox', 'OxygenTank']
        self.class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
    def load_best_model(self):
        """Load the best trained model"""
        model_paths = [
            "runs/train/simple_precision_boost/weights/best.pt",
            "runs/detect/train/weights/best.pt",
            "runs/detect/train2/weights/best.pt"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"✅ Loading model: {path}")
                return YOLO(path)
        
        print("❌ No trained model found!")
        return None
    
    def generate_confusion_matrix(self, model, test_data_path="yolo_params.yaml"):
        """Generate confusion matrix from model predictions"""
        print("🔍 Generating confusion matrix...")
        
        try:
            # Run validation to get detailed results
            results = model.val(data=test_data_path, split='test', save_json=True)
            
            # Create confusion matrix data
            confusion_data = []
            
            # Get predictions and ground truth
            for i, class_name in enumerate(self.class_names):
                # Get per-class metrics
                if hasattr(results.box, 'ap50') and i < len(results.box.ap50):
                    precision = results.box.ap50[i] if hasattr(results.box, 'ap50') else 0
                    recall = results.box.ap50[i] if hasattr(results.box, 'ap50') else 0
                    
                    confusion_data.append({
                        'Class': class_name,
                        'Precision': precision,
                        'Recall': recall,
                        'F1-Score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    })
            
            return pd.DataFrame(confusion_data)
            
        except Exception as e:
            print(f"❌ Error generating confusion matrix: {e}")
            return pd.DataFrame()
    
    def plot_confusion_matrix(self, confusion_df, save_path="confusion_matrix.png"):
        """Plot confusion matrix heatmap"""
        if confusion_df.empty:
            print("❌ No confusion matrix data available")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create correlation matrix for visualization
        metrics = ['Precision', 'Recall', 'F1-Score']
        correlation_matrix = confusion_df[metrics].T.corr()
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='Blues', 
                   fmt='.3f',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Correlation'})
        
        plt.title('Class Performance Correlation Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Classes', fontsize=12)
        plt.ylabel('Classes', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Confusion matrix saved to {save_path}")
    
    def plot_performance_metrics(self, confusion_df, save_path="performance_metrics.png"):
        """Plot performance metrics bar chart"""
        if confusion_df.empty:
            print("❌ No performance data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision comparison
        axes[0, 0].bar(confusion_df['Class'], confusion_df['Precision'], 
                      color=self.class_colors, alpha=0.8)
        axes[0, 0].set_title('Precision by Class', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(confusion_df['Precision']):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Recall comparison
        axes[0, 1].bar(confusion_df['Class'], confusion_df['Recall'], 
                      color=self.class_colors, alpha=0.8)
        axes[0, 1].set_title('Recall by Class', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(confusion_df['Recall']):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # F1-Score comparison
        axes[1, 0].bar(confusion_df['Class'], confusion_df['F1-Score'], 
                      color=self.class_colors, alpha=0.8)
        axes[1, 0].set_title('F1-Score by Class', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_ylim(0, 1)
        for i, v in enumerate(confusion_df['F1-Score']):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Overall performance radar chart
        metrics = ['Precision', 'Recall', 'F1-Score']
        avg_metrics = [confusion_df[metric].mean() for metric in metrics]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        avg_metrics += avg_metrics[:1]  # Close the plot
        angles += angles[:1]
        
        axes[1, 1].plot(angles, avg_metrics, 'o-', linewidth=2, color='red', alpha=0.7)
        axes[1, 1].fill(angles, avg_metrics, alpha=0.25, color='red')
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Average Performance Metrics', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Performance metrics saved to {save_path}")
    
    def plot_training_curves(self, results_paths, labels, save_path="training_curves.png"):
        """Plot training curves from multiple training runs"""
        plt.figure(figsize=(15, 10))
        
        metrics = [
            ('metrics/precision(B)', 'Precision'),
            ('metrics/recall(B)', 'Recall'),
            ('metrics/mAP50(B)', 'mAP@0.5'),
            ('metrics/mAP50-95(B)', 'mAP@0.5-0.95')
        ]
        
        for i, (metric_key, metric_name) in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            
            for results_path, label in zip(results_paths, labels):
                if os.path.exists(results_path):
                    try:
                        df = pd.read_csv(results_path)
                        if metric_key in df.columns:
                            plt.plot(df['epoch'], df[metric_key] * 100, 
                                   label=label, linewidth=2, marker='o', markersize=3)
                    except Exception as e:
                        print(f"❌ Error reading {results_path}: {e}")
            
            plt.title(f'{metric_name} Progress', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel(f'{metric_name} (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Training curves saved to {save_path}")
    
    def create_interactive_dashboard(self, confusion_df, save_path="performance_dashboard.html"):
        """Create interactive Plotly dashboard"""
        if confusion_df.empty:
            print("❌ No data for interactive dashboard")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Precision by Class', 'Recall by Class', 
                          'F1-Score by Class', 'Performance Radar'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatterpolar"}]]
        )
        
        # Bar charts
        fig.add_trace(
            go.Bar(x=confusion_df['Class'], y=confusion_df['Precision'],
                  name='Precision', marker_color=self.class_colors),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=confusion_df['Class'], y=confusion_df['Recall'],
                  name='Recall', marker_color=self.class_colors),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=confusion_df['Class'], y=confusion_df['F1-Score'],
                  name='F1-Score', marker_color=self.class_colors),
            row=2, col=1
        )
        
        # Radar chart
        metrics = ['Precision', 'Recall', 'F1-Score']
        avg_metrics = [confusion_df[metric].mean() for metric in metrics]
        
        fig.add_trace(
            go.Scatterpolar(
                r=avg_metrics + [avg_metrics[0]],
                theta=metrics + [metrics[0]],
                fill='toself',
                name='Average Performance',
                line_color='red'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Model Performance Dashboard",
            showlegend=False,
            height=800
        )
        
        # Save interactive dashboard
        fig.write_html(save_path)
        print(f"✅ Interactive dashboard saved to {save_path}")
    
    def generate_performance_report(self, confusion_df, save_path="performance_report.md"):
        """Generate comprehensive performance report"""
        if confusion_df.empty:
            print("❌ No data for performance report")
            return
        
        report = f"""
# Model Performance Report

## Summary
- **Overall Precision**: {confusion_df['Precision'].mean():.3f} ({confusion_df['Precision'].mean()*100:.1f}%)
- **Overall Recall**: {confusion_df['Recall'].mean():.3f} ({confusion_df['Recall'].mean()*100:.1f}%)
- **Overall F1-Score**: {confusion_df['F1-Score'].mean():.3f} ({confusion_df['F1-Score'].mean()*100:.1f}%)

## Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
"""
        
        for _, row in confusion_df.iterrows():
            report += f"| {row['Class']} | {row['Precision']:.3f} | {row['Recall']:.3f} | {row['F1-Score']:.3f} |\n"
        
        report += f"""
## Performance Analysis

### Best Performing Class
- **Class**: {confusion_df.loc[confusion_df['F1-Score'].idxmax(), 'Class']}
- **F1-Score**: {confusion_df['F1-Score'].max():.3f}

### Areas for Improvement
- **Lowest Precision**: {confusion_df.loc[confusion_df['Precision'].idxmin(), 'Class']} ({confusion_df['Precision'].min():.3f})
- **Lowest Recall**: {confusion_df.loc[confusion_df['Recall'].idxmin(), 'Class']} ({confusion_df['Recall'].min():.3f})

## Recommendations
1. **Model Performance**: {'Excellent' if confusion_df['F1-Score'].mean() > 0.9 else 'Good' if confusion_df['F1-Score'].mean() > 0.8 else 'Needs Improvement'}
2. **Precision Focus**: {'Achieved' if confusion_df['Precision'].mean() > 0.9 else 'Consider additional training'}
3. **Recall Balance**: {'Good balance' if abs(confusion_df['Precision'].mean() - confusion_df['Recall'].mean()) < 0.1 else 'Consider tuning'}

## Generated Files
- `confusion_matrix.png`: Class correlation matrix
- `performance_metrics.png`: Detailed performance charts
- `training_curves.png`: Training progress visualization
- `performance_dashboard.html`: Interactive dashboard
- `performance_report.md`: This report

---
*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
        """
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Performance report saved to {save_path}")

def main():
    print("🚀 COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    analyzer = PerformanceAnalyzer()
    
    # Load best model
    model = analyzer.load_best_model()
    if model is None:
        return
    
    # Generate confusion matrix
    confusion_df = analyzer.generate_confusion_matrix(model)
    
    if not confusion_df.empty:
        print("\n📊 Performance Metrics:")
        print(confusion_df.to_string(index=False))
        
        # Generate all visualizations
        print("\n🎨 Generating visualizations...")
        
        # Confusion matrix
        analyzer.plot_confusion_matrix(confusion_df)
        
        # Performance metrics
        analyzer.plot_performance_metrics(confusion_df)
        
        # Training curves
        results_paths = [
            "runs/train/simple_precision_boost/results.csv",
            "runs/detect/train/results.csv",
            "runs/detect/train2/results.csv"
        ]
        labels = ["Precision Boost", "Training Run 1", "Training Run 2"]
        analyzer.plot_training_curves(results_paths, labels)
        
        # Interactive dashboard
        analyzer.create_interactive_dashboard(confusion_df)
        
        # Performance report
        analyzer.generate_performance_report(confusion_df)
        
        print("\n✅ All performance analysis completed!")
        print("📁 Generated files:")
        print("   - confusion_matrix.png")
        print("   - performance_metrics.png")
        print("   - training_curves.png")
        print("   - performance_dashboard.html")
        print("   - performance_report.md")
        
    else:
        print("❌ Could not generate performance analysis")

if __name__ == "__main__":
    main() 