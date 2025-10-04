"""
Training Progress Monitor
Monitors training progress and displays key metrics
"""

import os
import time
import pandas as pd
from pathlib import Path

def monitor_training():
    """Monitor training progress in real-time"""
    
    print("🔍 TRAINING PROGRESS MONITOR")
    print("=" * 50)
    print("Press Ctrl+C to stop monitoring")
    print()
    
    # Training directories to monitor
    training_dirs = [
        "runs/train/precision_boost",
        "runs/train/high_precision_stage1", 
        "runs/train/high_precision_stage2"
    ]
    
    try:
        while True:
            for train_dir in training_dirs:
                results_file = os.path.join(train_dir, "results.csv")
                
                if os.path.exists(results_file):
                    try:
                        df = pd.read_csv(results_file)
                        if len(df) > 0:
                            latest = df.iloc[-1]
                            
                            print(f"📊 {train_dir.split('/')[-1].upper()}")
                            print(f"   Epoch: {latest['epoch']:.0f}")
                            print(f"   Precision: {latest['metrics/precision(B)']:.4f} ({latest['metrics/precision(B)']*100:.2f}%)")
                            print(f"   Recall: {latest['metrics/recall(B)']:.4f} ({latest['metrics/recall(B)']*100:.2f}%)")
                            print(f"   mAP@0.5: {latest['metrics/mAP50(B)']:.4f} ({latest['metrics/mAP50(B)']*100:.2f}%)")
                            print(f"   mAP@0.5-0.95: {latest['metrics/mAP50-95(B)']:.4f} ({latest['metrics/mAP50-95(B)']*100:.2f}%)")
                            
                            # Precision assessment
                            precision = latest['metrics/precision(B)']
                            if precision > 0.95:
                                print("   🎉 EXCELLENT! Precision > 95%")
                            elif precision > 0.90:
                                print("   ✅ GOOD! Precision > 90%")
                            else:
                                print("   ⚠️ Precision needs improvement")
                            
                            print()
                    except Exception as e:
                        print(f"❌ Error reading {results_file}: {e}")
            
            # Check for best model
            best_model_path = "runs/train/precision_boost/weights/best.pt"
            if os.path.exists(best_model_path):
                print("🏆 Best model found!")
                print(f"   Path: {best_model_path}")
                print()
            
            print("⏳ Waiting 30 seconds...")
            print("-" * 50)
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        print("💡 Use 'python check_map.py' for final results")

if __name__ == "__main__":
    monitor_training() 