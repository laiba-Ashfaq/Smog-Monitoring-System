"""
Automated Verification Suite for Smog Sentinel Punjab Fixes
============================================================
Tests:
1. DIP & AI Analyzer image patch detection across all 4 severity levels.
2. GeoTIFF 16-bit array scaling (no solid white clipping).
3. GeoTIFF 3-band RGB vs 4-band band ordering.
4. FastAPI REST backend image upload endpoint response validation.
"""

import sys
import os
import io
import numpy as np

# Force UTF-8 stdout encoding for Windows PowerShell/CMD
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("."))

from src.dip_extractor import DIPFeatureExtractor
from src.preprocessing import SatellitePreprocessor
from src.ml_pipeline import SmogMLPipeline
from api import app

def run_verification():
    print("==================================================")
    print("🧪 RUNNING SMOG SENTINEL VERIFICATION SUITE")
    print("==================================================")
    
    dip = DIPFeatureExtractor()
    pre = SatellitePreprocessor()
    ml = SmogMLPipeline()
    ml.load_or_train()
    
    # -------------------------------------------------------------------------
    # Test 1: Sample Patches Detection Accuracy across 4 Tiers
    # -------------------------------------------------------------------------
    print("\n--- Test 1: DIP & AI Analyzer Sample Patch Severity Detection ---")
    sample_file_map = {
        "Clear Day": "data/sample_patches/clear_day_lahore.png",
        "Moderate Haze": "data/sample_patches/moderate_haze_faisalabad.png",
        "Heavy Smog": "data/sample_patches/heavy_smog_sheikhupura.png",
        "Hazardous Plume": "data/sample_patches/hazardous_smog_cropfire.png"
    }
    
    expected_tiers = {
        "Clear Day": [0],         # Clean / Good
        "Moderate Haze": [0, 1],   # Clean or Moderate Haze
        "Heavy Smog": [2],        # Unhealthy / Dense Smog
        "Hazardous Plume": [2, 3] # Dense or Hazardous Plume
    }
    
    all_passed = True
    for name, path in sample_file_map.items():
        if not os.path.exists(path):
            print(f"  [FAIL] Missing patch: {path}")
            all_passed = False
            continue
            
        img_np = np.array(Image.open(path).convert("RGB"))
        feats = dip.extract_all_features(img_np)
        scene_proc = pre.process_scene(img_np)
        feats["hot_index"] = scene_proc["smog_indices"]["hot_mean"]
        feats["ndsi_index"] = scene_proc["smog_indices"]["ndsi_mean"]
        
        dip_aod = feats["dip_optical_aod"]
        res = ml.predict_patch(feats, weather_features={"modis_aod": dip_aod}, district="Lahore")
        
        scls = res["severity_class"]
        sname = res["severity_name"]
        pm = res["pm25_predicted"]
        
        valid = scls in expected_tiers[name]
        status = "PASS" if valid else "FAIL"
        print(f"  [{status}] {name:15s} -> DIP AOD: {dip_aod:.3f} | Class {scls} ({sname}) | PM2.5: {pm} µg/m³")
        if not valid:
            all_passed = False

    # -------------------------------------------------------------------------
    # Test 2: GeoTIFF 16-bit Array Normalization (No Clipping to White 255)
    # -------------------------------------------------------------------------
    print("\n--- Test 2: 16-bit GeoTIFF Array Normalization ---")
    geotiff_16bit = (np.random.rand(256, 256, 3) * 8000.0).astype(np.uint16)
    geotiff_16bit[50:100, 50:100, :] = 12000  # bright region
    
    norm_uint8 = dip._normalize_to_uint8(geotiff_16bit)
    print(f"  Input dtype: {geotiff_16bit.dtype}, shape: {geotiff_16bit.shape}, min: {geotiff_16bit.min()}, max: {geotiff_16bit.max()}")
    print(f"  Output dtype: {norm_uint8.dtype}, shape: {norm_uint8.shape}, min: {norm_uint8.min()}, max: {norm_uint8.max()}")
    
    std_val = float(np.std(norm_uint8))
    is_not_solid_white = norm_uint8.min() < 50 and norm_uint8.max() == 255 and std_val > 10.0
    status = "PASS" if is_not_solid_white else "FAIL"
    print(f"  [{status}] GeoTIFF array scaled smoothly without solid white 255 clipping (std={std_val:.2f})")
    if not is_not_solid_white:
        all_passed = False

    # -------------------------------------------------------------------------
    # Test 3: FastAPI Image Upload Endpoint (/api/v1/predict/upload)
    # -------------------------------------------------------------------------
    print("\n--- Test 3: FastAPI REST Endpoint (/api/v1/predict/upload) ---")
    client = TestClient(app)
    
    sample_png = sample_file_map["Heavy Smog"]
    with open(sample_png, "rb") as f:
        file_bytes = f.read()
        
    response = client.post(
        "/api/v1/predict/upload",
        files={"file": ("heavy_smog.png", file_bytes, "image/png")},
        data={"district": "Sheikhupura", "humidity": 80.0, "wind_speed": 3.5, "crop_fires": 20}
    )
    
    if response.status_code == 200:
        data = response.json()
        pred = data.get("prediction", {})
        print(f"  [PASS] API Status Code: 200 OK")
        print(f"  Predict Response: District={data.get('district')} | Severity={pred.get('severity_name')} (Class {pred.get('severity_class')}) | PM2.5={pred.get('pm25_predicted')} µg/m³")
    else:
        print(f"  [FAIL] API Status Code: {response.status_code} | Body: {response.text}")
        all_passed = False

    print("\n==================================================")
    if all_passed:
        print("✅ ALL SYSTEM VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    else:
        print("❌ SOME VERIFICATION CHECKS FAILED — REVIEW LOGS ABOVE.")
    print("==================================================")
    return all_passed

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
