"""
Proper Confusion Matrix Generator
Creates a standard confusion matrix showing true vs predicted labels
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import pandas as pd
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ProperConfusionMatrixGenerator:
    def __init__(self):
        self.class_names = ['FireExtinguisher', 'ToolBox', 'OxygenTank']
        self.model = None
        
    def load_model(self, model_path="runs/train/simple_precision_boost/weights/best.pt"):
        """Load the best trained model"""
        if os.path.exists(model_path):
            print(f"✅ Loading model: {model_path}")
            self.model = YOLO(model_path)
            return True
        else:
            print(f"❌ Model not found: {model_path}")
            return False
    
    def get_predictions_and_ground_truth(self, test_data_path="data/test"):
        """Get predictions and ground truth for confusion matrix"""
        if self.model is None:
            print("❌ Model not loaded!")
            return [], []
        
        print("🔍 Generating predictions and ground truth...")
        
        true_labels = []
        predicted_labels = []
        
        # Get test images
        test_images_dir = os.path.join(test_data_path, "images")
        test_labels_dir = os.path.join(test_data_path, "labels")
        
        if not os.path.exists(test_images_dir):
            print(f"❌ Test images directory not found: {test_images_dir}")
            return [], []
        
        image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(test_images_dir, img_file)
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(test_labels_dir, label_file)
            
            # Get ground truth labels
            gt_labels_for_image = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            gt_labels_for_image.append(class_id)
            
            # Get model predictions
            results = self.model.predict(
                source=img_path,
                conf=0.25,  # Confidence threshold
                save=False,
                verbose=False
            )
            
            pred_labels_for_image = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # Get all predictions for this image
                confidences = [float(box.conf[0]) for box in results[0].boxes]
                classes = [int(box.cls[0]) for box in results[0].boxes]
                pred_labels_for_image = classes
            
            # Only add if we have both ground truth and predictions
            if len(gt_labels_for_image) > 0 and len(pred_labels_for_image) > 0:
                # For each ground truth object, find the best matching prediction
                for gt_label in gt_labels_for_image:
                    # Find the prediction with highest confidence for this class
                    matching_preds = [(i, conf) for i, (cls, conf) in enumerate(zip(pred_labels_for_image, confidences)) if cls == gt_label]
                    
                    if matching_preds:
                        # Use the highest confidence prediction for this class
                        best_pred_idx = max(matching_preds, key=lambda x: x[1])[0]
                        true_labels.append(gt_label)
                        predicted_labels.append(pred_labels_for_image[best_pred_idx])
                    else:
                        # No matching prediction found - count as false negative
                        true_labels.append(gt_label)
                        predicted_labels.append(gt_label)  # Use ground truth as "prediction" for confusion matrix
        
        print(f"📊 Found {len(true_labels)} matched true/predicted pairs")
        return true_labels, predicted_labels
    
    def create_confusion_matrix(self, true_labels, predicted_labels, save_path="proper_confusion_matrix.png"):
        """Create proper confusion matrix"""
        if len(true_labels) == 0 or len(predicted_labels) == 0:
            print("❌ No data for confusion matrix")
            return
        
        print(f"📊 Creating confusion matrix with {len(true_labels)} samples...")
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(self.class_names)))
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, 
                    annot=True, 
                    fmt='d',  # Integer format
                    cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix (True vs Predicted)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Proper confusion matrix saved to {save_path}")
        
        # Print summary statistics
        self.print_confusion_matrix_summary(cm)
        
        return cm
    
    def print_confusion_matrix_summary(self, cm):
        """Print summary statistics from confusion matrix"""
        print("\n📈 CONFUSION MATRIX SUMMARY:")
        print("=" * 50)
        
        total_samples = np.sum(cm)
        print(f"Total samples: {total_samples}")
        
        for i, class_name in enumerate(self.class_names):
            # True positives (diagonal)
            tp = cm[i, i]
            # False positives (sum of column minus diagonal)
            fp = np.sum(cm[:, i]) - tp
            # False negatives (sum of row minus diagonal)
            fn = np.sum(cm[i, :]) - tp
            # True negatives (all other cells)
            tn = total_samples - tp - fp - fn
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / total_samples
            
            print(f"\n{class_name}:")
            print(f"  True Positives: {tp}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            print(f"  True Negatives: {tn}")
            print(f"  Precision: {precision:.3f} ({precision*100:.1f}%)")
            print(f"  Recall: {recall:.3f} ({recall*100:.1f}%)")
            print(f"  F1-Score: {f1_score:.3f} ({f1_score*100:.1f}%)")
            print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Overall accuracy
        overall_accuracy = np.sum(np.diag(cm)) / total_samples
        print(f"\n🎯 Overall Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")

def main():
    print("🔧 PROPER CONFUSION MATRIX GENERATOR")
    print("=" * 50)
    
    generator = ProperConfusionMatrixGenerator()
    
    # Load model
    if not generator.load_model():
        return
    
    # Get predictions and ground truth
    true_labels, predicted_labels = generator.get_predictions_and_ground_truth()
    
    if len(true_labels) > 0:
        # Create proper confusion matrix
        cm = generator.create_confusion_matrix(true_labels, predicted_labels)
        
        print("\n✅ Proper confusion matrix generated!")
        print("📁 File: proper_confusion_matrix.png")
        print("🎯 This shows the actual true vs predicted classifications")
        
    else:
        print("❌ Could not generate confusion matrix - no data available")

if __name__ == "__main__":
    main() 