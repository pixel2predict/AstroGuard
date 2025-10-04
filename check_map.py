import pandas as pd
import os

def check_current_map():
    """Check current mAP scores from training results"""
    
    # Check for training results
    train_results = "runs/detect/train/results.csv"
    train2_results = "runs/detect/train2/results.csv"
    
    print("=== CURRENT MODEL mAP SCORES ===\n")
    
    if os.path.exists(train_results):
        print("📊 TRAINING RUN 1 RESULTS:")
        df = pd.read_csv(train_results)
        
        # Get the best epoch (highest mAP50-95)
        best_epoch = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
        
        print(f"   Best Epoch: {best_epoch['epoch']:.0f}")
        print(f"   mAP@0.5: {best_epoch['metrics/mAP50(B)']:.4f} ({best_epoch['metrics/mAP50(B)']*100:.2f}%)")
        print(f"   mAP@0.5-0.95: {best_epoch['metrics/mAP50-95(B)']:.4f} ({best_epoch['metrics/mAP50-95(B)']*100:.2f}%)")
        print(f"   Precision: {best_epoch['metrics/precision(B)']:.4f} ({best_epoch['metrics/precision(B)']*100:.2f}%)")
        print(f"   Recall: {best_epoch['metrics/recall(B)']:.4f} ({best_epoch['metrics/recall(B)']*100:.2f}%)")
        
        # Get final epoch results
        final_epoch = df.iloc[-1]
        print(f"\n   Final Epoch ({final_epoch['epoch']:.0f}):")
        print(f"   mAP@0.5: {final_epoch['metrics/mAP50(B)']:.4f} ({final_epoch['metrics/mAP50(B)']*100:.2f}%)")
        print(f"   mAP@0.5-0.95: {final_epoch['metrics/mAP50-95(B)']:.4f} ({final_epoch['metrics/mAP50-95(B)']*100:.2f}%)")
        print(f"   Precision: {final_epoch['metrics/precision(B)']:.4f} ({final_epoch['metrics/precision(B)']*100:.2f}%)")
        print(f"   Recall: {final_epoch['metrics/recall(B)']:.4f} ({final_epoch['metrics/recall(B)']*100:.2f}%)")
        
    else:
        print("❌ No training results found in runs/detect/train/")
    
    print("\n" + "="*50 + "\n")
    
    if os.path.exists(train2_results):
        print("📊 TRAINING RUN 2 RESULTS:")
        df2 = pd.read_csv(train2_results)
        
        # Get the best epoch (highest mAP50-95)
        if len(df2) > 0:
            best_epoch2 = df2.loc[df2['metrics/mAP50-95(B)'].idxmax()]
            
            print(f"   Best Epoch: {best_epoch2['epoch']:.0f}")
            print(f"   mAP@0.5: {best_epoch2['metrics/mAP50(B)']:.4f} ({best_epoch2['metrics/mAP50(B)']*100:.2f}%)")
            print(f"   mAP@0.5-0.95: {best_epoch2['metrics/mAP50-95(B)']:.4f} ({best_epoch2['metrics/mAP50-95(B)']*100:.2f}%)")
            print(f"   Precision: {best_epoch2['metrics/precision(B)']:.4f} ({best_epoch2['metrics/precision(B)']*100:.2f}%)")
            print(f"   Recall: {best_epoch2['metrics/recall(B)']:.4f} ({best_epoch2['metrics/recall(B)']*100:.2f}%)")
            
            # Get final epoch results
            final_epoch2 = df2.iloc[-1]
            print(f"\n   Final Epoch ({final_epoch2['epoch']:.0f}):")
            print(f"   mAP@0.5: {final_epoch2['metrics/mAP50(B)']:.4f} ({final_epoch2['metrics/mAP50(B)']*100:.2f}%)")
            print(f"   mAP@0.5-0.95: {final_epoch2['metrics/mAP50-95(B)']:.4f} ({final_epoch2['metrics/mAP50-95(B)']*100:.2f}%)")
            print(f"   Precision: {final_epoch2['metrics/precision(B)']:.4f} ({final_epoch2['metrics/precision(B)']*100:.2f}%)")
            print(f"   Recall: {final_epoch2['metrics/recall(B)']:.4f} ({final_epoch2['metrics/recall(B)']*100:.2f}%)")
        else:
            print("   No valid results found")
    else:
        print("❌ No training results found in runs/detect/train2/")
    
    print("\n" + "="*50)
    print("\n💡 RECOMMENDATIONS:")
    print("1. The best model shows good performance but could be improved")
    print("2. Consider running the improved training script for better results")
    print("3. Use train_improved.py for enhanced hyperparameters and longer training")

if __name__ == "__main__":
    check_current_map() 