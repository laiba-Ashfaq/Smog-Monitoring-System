"""
Training and Sample Generation Runner
"""
import sys
import os
# Ensure current directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.sample_generator import generate_sample_patches
from src.ml_pipeline import SmogMLPipeline

if __name__ == "__main__":
    print("[1/2] Generating sample satellite image patches...")
    generate_sample_patches()

    print("[2/2] Generating calibrated dataset and training ML models...")
    pipeline = SmogMLPipeline()
    metrics = pipeline.train_models()

    print("--------------------------------------------------")
    print("[SUCCESS] Training and Leak-Free Validation Complete!")
    print(f"Split Method: {metrics.get('split_method', 'unknown')}")
    print(f"Group Overlap: {metrics.get('group_overlap_count', 'n/a')} "
          f"({'Strict Zero Leakage' if metrics.get('leak_free_verified') else 'LEAKAGE DETECTED - CHECK SPLIT'})")

    cls_test = metrics.get('classification_accuracy')
    cls_train = metrics.get('classification_accuracy_train')
    cls_gap = metrics.get('classification_accuracy_gap')
    r2_test = metrics.get('r2_score')
    r2_train = metrics.get('r2_score_train')
    r2_gap = metrics.get('r2_score_gap')

    print(f"Classification Accuracy — Test: {cls_test*100:.2f}%  Train: {cls_train*100:.2f}%  Gap: {cls_gap*100:.2f} pts")
    print(f"R2 Score — Test: {r2_test:.4f}  Train: {r2_train:.4f}  Gap: {r2_gap:.4f}")
    print(f"Smog Area IoU (>=Moderate): {metrics.get('iou_smog_area_ge1', metrics.get('iou_smog_detection')):.3f}")
    print(f"Dense Smog IoU (>=Unhealthy): {metrics.get('iou_dense_smog_ge2', float('nan')):.3f}")
    print(f"MAE: {metrics.get('mae_ug_m3')} ug/m3")
    print(f"RMSE: {metrics.get('rmse_ug_m3')} ug/m3")

    cls_overfit = metrics.get('classification_overfit_flag', False)
    reg_overfit = metrics.get('regression_overfit_flag', False)
    if cls_overfit or reg_overfit:
        print(f"Overfitting Audit: FAILED"
              f"{' (classifier gap > 4%)' if cls_overfit else ''}"
              f"{' (regressor gap > 0.15 R2)' if reg_overfit else ''}")
    else:
        print("Overfitting Audit: PASSED")
    print("--------------------------------------------------")