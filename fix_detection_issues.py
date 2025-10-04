"""
Detection Issues Fix Script
Helps identify and fix issues with images that aren't being detected properly
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

class DetectionFixer:
    def __init__(self):
        self.model = None
        self.class_names = ['FireExtinguisher', 'ToolBox', 'OxygenTank']
        
    def load_model(self, model_path="runs/train/simple_precision_boost/weights/best.pt"):
        """Load the best trained model"""
        if os.path.exists(model_path):
            print(f"✅ Loading model: {model_path}")
            self.model = YOLO(model_path)
            return True
        else:
            print(f"❌ Model not found: {model_path}")
            return False
    
    def analyze_image_detection(self, image_path, conf_thresholds=[0.1, 0.25, 0.5, 0.75]):
        """Analyze detection at different confidence thresholds"""
        if self.model is None:
            print("❌ Model not loaded!")
            return
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"🔍 Analyzing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Could not load image!")
            return
        
        # Test different confidence thresholds
        results_summary = []
        
        for conf in conf_thresholds:
            results = self.model.predict(
                source=image,
                conf=conf,
                save=False,
                verbose=False
            )
            
            detections = len(results[0].boxes) if results[0].boxes is not None else 0
            max_conf = max([float(box.conf[0]) for box in results[0].boxes]) if detections > 0 else 0
            
            results_summary.append({
                'Confidence_Threshold': conf,
                'Detections': detections,
                'Max_Confidence': max_conf,
                'Classes_Detected': [self.class_names[int(box.cls[0])] for box in results[0].boxes] if detections > 0 else []
            })
        
        return pd.DataFrame(results_summary)
    
    def suggest_confidence_threshold(self, image_path):
        """Suggest optimal confidence threshold for an image"""
        df = self.analyze_image_detection(image_path)
        
        if df.empty:
            return None
        
        print(f"\n📊 Detection Analysis for {os.path.basename(image_path)}:")
        print(df.to_string(index=False))
        
        # Find best threshold
        if df['Detections'].max() > 0:
            best_row = df.loc[df['Detections'].idxmax()]
            suggested_conf = best_row['Confidence_Threshold']
            
            print(f"\n💡 Suggested confidence threshold: {suggested_conf}")
            print(f"   Detections: {best_row['Detections']}")
            print(f"   Classes: {', '.join(best_row['Classes_Detected'])}")
            
            return suggested_conf
        else:
            print(f"\n⚠️ No detections found at any confidence level!")
            print("   Possible issues:")
            print("   - Image doesn't contain target objects")
            print("   - Image quality is too low")
            print("   - Objects are too small or occluded")
            print("   - Lighting conditions are poor")
            
            return None
    
    def test_image_preprocessing(self, image_path):
        """Test different image preprocessing techniques"""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"🔄 Testing image preprocessing for: {image_path}")
        
        # Load original image
        original = cv2.imread(image_path)
        
        # Different preprocessing techniques
        preprocessing_methods = {
            'Original': original,
            'Brightened': cv2.convertScaleAbs(original, alpha=1.3, beta=30),
            'Contrast_Enhanced': cv2.convertScaleAbs(original, alpha=1.5, beta=0),
            'Blur_Reduced': cv2.GaussianBlur(original, (3, 3), 0),
            'Sharpened': cv2.filter2D(original, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
        }
        
        results = []
        
        for method_name, processed_image in preprocessing_methods.items():
            if self.model is None:
                continue
                
            results_pred = self.model.predict(
                source=processed_image,
                conf=0.25,
                save=False,
                verbose=False
            )
            
            detections = len(results_pred[0].boxes) if results_pred[0].boxes is not None else 0
            results.append({
                'Method': method_name,
                'Detections': detections,
                'Classes': [self.class_names[int(box.cls[0])] for box in results_pred[0].boxes] if detections > 0 else []
            })
        
        return pd.DataFrame(results)
    
    def create_detection_report(self, image_folder="data/test/images", sample_size=10):
        """Create a comprehensive detection report for multiple images"""
        if not os.path.exists(image_folder):
            print(f"❌ Image folder not found: {image_folder}")
            return
        
        print(f"📋 Creating detection report for {image_folder}")
        
        # Get sample images
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        sample_images = image_files[:sample_size]
        
        report_data = []
        
        for img_file in sample_images:
            img_path = os.path.join(image_folder, img_file)
            
            # Test detection
            if self.model is None:
                continue
                
            results = self.model.predict(
                source=img_path,
                conf=0.25,
                save=False,
                verbose=False
            )
            
            detections = len(results[0].boxes) if results[0].boxes is not None else 0
            classes_detected = [self.class_names[int(box.cls[0])] for box in results[0].boxes] if detections > 0 else []
            
            report_data.append({
                'Image': img_file,
                'Detections': detections,
                'Classes': ', '.join(classes_detected) if classes_detected else 'None',
                'Status': '✅ Detected' if detections > 0 else '❌ No Detection'
            })
        
        df = pd.DataFrame(report_data)
        
        print(f"\n📊 Detection Report (Sample of {len(sample_images)} images):")
        print(df.to_string(index=False))
        
        # Summary statistics
        total_images = len(df)
        detected_images = len(df[df['Detections'] > 0])
        detection_rate = (detected_images / total_images) * 100
        
        print(f"\n📈 Summary:")
        print(f"   Total Images: {total_images}")
        print(f"   Images with Detections: {detected_images}")
        print(f"   Detection Rate: {detection_rate:.1f}%")
        
        if detection_rate < 80:
            print(f"\n⚠️ Low detection rate detected!")
            print("   Recommendations:")
            print("   - Lower confidence threshold")
            print("   - Check image quality")
            print("   - Verify object presence in images")
            print("   - Consider retraining with more diverse data")
        
        return df

def main():
    print("🔧 DETECTION ISSUES FIXER")
    print("=" * 40)
    
    fixer = DetectionFixer()
    
    # Load model
    if not fixer.load_model():
        return
    
    # Test with sample images
    test_images = [
        "data/test/images/000000000.png",
        "data/test/images/000000001.png",
        "data/test/images/000000002.png"
    ]
    
    print("\n🔍 Testing sample images...")
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n--- Testing {os.path.basename(img_path)} ---")
            
            # Analyze detection
            fixer.suggest_confidence_threshold(img_path)
            
            # Test preprocessing
            preprocessing_results = fixer.test_image_preprocessing(img_path)
            if not preprocessing_results.empty:
                print(f"\n🔄 Preprocessing Results:")
                print(preprocessing_results.to_string(index=False))
    
    # Create comprehensive report
    print(f"\n📋 Creating comprehensive detection report...")
    fixer.create_detection_report()
    
    print(f"\n✅ Detection analysis completed!")
    print(f"💡 Tips for improving detection:")
    print(f"   - Try different confidence thresholds")
    print(f"   - Use image preprocessing techniques")
    print(f"   - Check image quality and lighting")
    print(f"   - Ensure objects are clearly visible")

if __name__ == "__main__":
    main() 