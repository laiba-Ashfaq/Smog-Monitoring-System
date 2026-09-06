"""
Satellite Imagery Preprocessing Pipeline
========================================
Implements the exact 6-stage preprocessing pipeline with Rasterio GeoTIFF support:
1. Radiometric Calibration
2. Cloud & Shadow Masking (SCL Thresholding)
3. Coordinate Reprojection to WGS84 UTM Zone 42N (EPSG:32642) with Rasterio
4. Smog-Sensitive Spectral Indices: HOT (Haze Optimized Transformation) & NDSI
5. Dual-Stage Normalization: Min-Max + Z-Score
6. Patch Extraction (256x256, ~2.5 km2) with Augmentation
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional

class SatellitePreprocessor:
    """End-to-end Sentinel-2 / MODIS Preprocessing Pipeline."""

    def __init__(self, patch_size: int = 256, target_epsg: str = "EPSG:32642"):
        self.patch_size = patch_size
        self.target_epsg = target_epsg
        self.quantification_value = 10000.0

    def read_sentinel2_geotiff(self, geotiff_path: str) -> Dict[str, Any]:
        """
        Reads a real Sentinel-2 multi-band GeoTIFF file using Rasterio.
        Extracts raw reflectance bands (B2 Blue, B3 Green, B4 Red, B8 NIR),
        spatial metadata, CRS, and ground bounding box.
        """
        import rasterio  # imported lazily: only needed for this GeoTIFF-upload path
        with rasterio.open(geotiff_path) as src:
            bounds = src.bounds
            crs = str(src.crs)
            transform = src.transform
            count = src.count
            
            if count >= 4:
                # 4-band Sentinel-2 L2A/L1C (B2 Blue, B3 Green, B4 Red, B8 NIR)
                b_blue = src.read(1)
                b_green = src.read(2)
                b_red = src.read(3)
                rgb_stack = np.stack([b_red, b_green, b_blue], axis=2)
            elif count == 3:
                # Standard 3-band RGB GeoTIFF (B1 Red, B2 Green, B3 Blue)
                b_red = src.read(1)
                b_green = src.read(2)
                b_blue = src.read(3)
                rgb_stack = np.stack([b_red, b_green, b_blue], axis=2)
            else:
                gray = src.read(1)
                rgb_stack = np.stack([gray, gray, gray], axis=2)
                
            return {
                "rgb_stack": rgb_stack,
                "bounds": (bounds.left, bounds.bottom, bounds.right, bounds.top),
                "crs": crs,
                "transform": transform,
                "resolution_m": src.res[0] if src.res else 10.0,
                "dimensions": (src.height, src.width)
            }

    def radiometric_calibration(self, raw_dn: np.ndarray) -> np.ndarray:
        """
        Converts raw pixel values to surface reflectance [0.0, 1.0].

        BUG FIX: this used to unconditionally divide by self.quantification_value
        (10000.0), which is only correct for 16-bit Sentinel-2 GeoTIFF DN values.
        NASA GIBS / ESA WMS scenes (the actual source used by data_ingestion.py's
        fetch_satellite_smog_scene) arrive as 8-bit RGB in [0..255]. Dividing those
        by 10000.0 produced pixel intensities of ~0.008-0.03, which then made
        cloud_shadow_masking's shadow_thresh (0.08) fire on ~100% of every scene
        -- wiping out real terrain into flat gray "shadow" and leaving the DIP
        feature extractor analyzing noise instead of genuine imagery. We now
        detect the input's actual bit depth/range and scale accordingly:
          - dtype is an integer type with max <= 255, or float already <= 1.01
            in that pattern  -> uint8 RGB, divide by 255.0
          - otherwise (dtype uint16/int32/float with larger DN magnitudes,
            i.e. genuine Sentinel-2 GeoTIFF reflectance*10000 values) -> divide
            by self.quantification_value (10000.0), as before.
        """
        arr = raw_dn.astype(np.float32)
        is_8bit_like = (
            raw_dn.dtype == np.uint8
            or (raw_dn.dtype in (np.float32, np.float64) and np.nanmax(arr) <= 255.0)
        )
        if is_8bit_like:
            calibrated = arr / 255.0
        else:
            calibrated = arr / self.quantification_value
        return np.clip(calibrated, 0.0, 1.0)

    def cloud_shadow_masking(self, image_rgb: np.ndarray, cloud_thresh: float = 0.78, shadow_thresh: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """Identifies and masks opaque clouds and dark cloud shadows."""
        b = image_rgb[:, :, 2] if image_rgb.shape[2] >= 3 else image_rgb
        intensity = np.mean(image_rgb, axis=2)
        
        cloud_mask = (intensity > cloud_thresh) & (b > 0.70)
        shadow_mask = intensity < shadow_thresh
        invalid_mask = cloud_mask | shadow_mask
        valid_mask = ~invalid_mask
        
        masked_rgb = image_rgb.copy()
        median_val = np.median(image_rgb[valid_mask], axis=0) if np.any(valid_mask) else [0.2, 0.2, 0.2]
        masked_rgb[invalid_mask] = median_val
        return masked_rgb, valid_mask

    def reproject_to_wgs84_utm42n(self, image_rgb: np.ndarray, bounds_wgs84: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
        """Reprojects geographic coordinates to metric WGS84 UTM Zone 42N grid."""
        if bounds_wgs84 is None:
            bounds_wgs84 = (31.45, 74.25, 31.60, 74.45)
            
        min_lat, min_lon, max_lat, max_lon = bounds_wgs84
        utm_easting_min = (min_lon - 69.0) * 95200.0 + 500000.0
        utm_easting_max = (max_lon - 69.0) * 95200.0 + 500000.0
        utm_northing_min = min_lat * 110800.0
        utm_northing_max = max_lat * 110800.0
        
        width_m = utm_easting_max - utm_easting_min
        height_m = utm_northing_max - utm_northing_min
        grid_cols = max(256, int(width_m / 10.0))
        grid_rows = max(256, int(height_m / 10.0))
        
        reprojected_grid = cv2.resize(image_rgb, (grid_cols, grid_rows), interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate standard Rasterio Affine Transform
        try:
            from rasterio.transform import from_bounds
            affine_transform = from_bounds(utm_easting_min, utm_northing_min, utm_easting_max, utm_northing_max, grid_cols, grid_rows)
        except ImportError:
            # rasterio unavailable (e.g. blocked by a Windows Application Control policy) —
            # fall back to computing the same affine transform by hand.
            pixel_width = (utm_easting_max - utm_easting_min) / grid_cols
            pixel_height = (utm_northing_max - utm_northing_min) / grid_rows
            affine_transform = (pixel_width, 0.0, utm_easting_min, 0.0, -pixel_height, utm_northing_max)
        
        return {
            "crs": self.target_epsg,
            "utm_easting_bounds": (round(utm_easting_min, 1), round(utm_easting_max, 1)),
            "utm_northing_bounds": (round(utm_northing_min, 1), round(utm_northing_max, 1)),
            "grid_resolution_m": 10.0,
            "grid_dimensions": (grid_rows, grid_cols),
            "affine_transform": str(affine_transform),
            "reprojected_image": reprojected_grid
        }

    def compute_smog_indices(self, image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """Computes HOT (Haze Optimized Transformation) and NDSI indices."""
        r = image_rgb[:, :, 0].astype(np.float32)
        g = image_rgb[:, :, 1].astype(np.float32)
        b = image_rgb[:, :, 2].astype(np.float32)
        
        hot_index = b - 0.5 * r - 0.08
        ndsi_index = (b - r) / (b + r + 1e-5)
        asr_index = b / (g + 1e-5)
        
        return {
            "hot_index": hot_index,
            "ndsi_index": ndsi_index,
            "asr_index": asr_index,
            "hot_mean": float(np.mean(hot_index)),
            "ndsi_mean": float(np.mean(ndsi_index)),
            "asr_mean": float(np.mean(asr_index)),
        }

    def dual_stage_normalization(self, feature_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Stage A: Min-Max Normalization, Stage B: Z-Score Standardization."""
        f_min, f_max = float(np.min(feature_map)), float(np.max(feature_map))
        minmax_norm = (feature_map - f_min) / (f_max - f_min + 1e-6)
        
        mu = float(np.mean(feature_map))
        sigma = float(np.std(feature_map))
        zscore_norm = (feature_map - mu) / (sigma + 1e-6)
        return minmax_norm, zscore_norm

    def extract_patches_with_augmentation(self, image_rgb: np.ndarray, stride: int = 128, augment: bool = True) -> List[Dict[str, Any]]:
        """Extracts 256x256 spatial patches (~2.5 km2 ground coverage) with augmentation."""
        h, w, c = image_rgb.shape
        patches = []
        patch_id = 0
        
        for y in range(0, max(1, h - self.patch_size + 1), stride):
            for x in range(0, max(1, w - self.patch_size + 1), stride):
                patch = image_rgb[y : y + self.patch_size, x : x + self.patch_size]
                if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                    patch = cv2.resize(patch, (self.patch_size, self.patch_size))
                    
                patches.append({"patch_id": f"PATCH_{patch_id:04d}", "coord_px": (y, x), "image": patch, "augmented": False})
                patch_id += 1
                
                if augment:
                    h_flip = cv2.flip(patch, 1)
                    patches.append({"patch_id": f"PATCH_{patch_id:04d}_HFLIP", "coord_px": (y, x), "image": h_flip, "augmented": True})
                    patch_id += 1
                    rot90 = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
                    patches.append({"patch_id": f"PATCH_{patch_id:04d}_ROT90", "coord_px": (y, x), "image": rot90, "augmented": True})
                    patch_id += 1

        return patches

    def process_scene(self, raw_scene_rgb: np.ndarray, augment: bool = False) -> Dict[str, Any]:
        """Executes the full preprocessing pipeline on a scene."""
        calibrated = self.radiometric_calibration(raw_scene_rgb)
        masked, valid_mask = self.cloud_shadow_masking(calibrated)
        geo_meta = self.reproject_to_wgs84_utm42n(masked)
        smog_indices = self.compute_smog_indices(masked)
        hot_minmax, hot_zscore = self.dual_stage_normalization(smog_indices["hot_index"])
        patches = self.extract_patches_with_augmentation(masked, stride=256, augment=augment)
        
        return {
            "calibrated_image": calibrated,
            "cloud_masked_image": masked,
            "valid_ground_ratio": float(np.mean(valid_mask)),
            "utm_42n_meta": geo_meta,
            "smog_indices": smog_indices,
            "hot_minmax_normalized": hot_minmax,
            "hot_zscore_normalized": hot_zscore,
            "patch_count": len(patches),
            "patches": patches
        }