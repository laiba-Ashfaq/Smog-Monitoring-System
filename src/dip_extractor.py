"""
Digital Image Processing (DIP) Feature Extraction Module
========================================================
Implements classical DIP techniques to quantify smog severity from satellite/aerial/CCTV imagery:
1. Blue Channel Histogram Analysis (Haze / Mie scattering)
2. Edge Sharpness & Gradient Metric (Aerosol edge attenuation)
3. HSI / HSV Color Space Saturation Degradation (Purity loss)
4. 2D Fast Fourier Transform (FFT) High-Frequency Energy Ratio
"""

import numpy as np
import cv2
from scipy import stats
from scipy.fftpack import fft2, fftshift
from typing import Dict, Tuple, Any

class DIPFeatureExtractor:
    """Extracts classical DIP features from satellite, aerial, or ground camera patches."""

    def __init__(self, high_freq_radius_ratio: float = 0.25):
        """
        Args:
            high_freq_radius_ratio: Normalized radius cutoff in FFT frequency plane.
        """
        self.high_freq_radius_ratio = high_freq_radius_ratio

    def extract_blue_histogram_features(self, image_rgb: np.ndarray) -> Dict[str, float]:
        """
        1. Blue Channel Histogram Analysis:
        Smoggy atmosphere scatters shorter wavelengths (Mie/Rayleigh scattering),
        causing blue channel histograms to become right-shifted, brighter, and narrower.
        """
        blue_channel = image_rgb[:, :, 2].astype(np.float32)  # RGB -> Blue is index 2
        mean_val = float(np.mean(blue_channel))
        std_val = float(np.std(blue_channel))
        
        # Calculate skewness (measure of asymmetry in distribution)
        flattened = blue_channel.flatten()
        skew_val = float(stats.skew(flattened)) if std_val > 1e-6 else 0.0
        
        # High intensity blue fraction. NOTE: a fixed absolute cutoff has broken twice now
        # for the same underlying reason -- it silently goes constant (zero variance for
        # every sample) whenever an upstream calibration change shifts the pipeline's
        # actual blue-channel range, which just happened again after the radiometric
        # calibration bug fix changed patch intensities. diagnose_group_leakage.py
        # confirmed blue_high_ratio had ZERO variance across the entire real dataset
        # (not just within-day) -- a dead feature, unrelated to the day-identity/ICC
        # issue affecting other features. Using a threshold relative to THIS patch's own
        # mean+std keeps the "fraction of haze-brightened blue pixels" semantics while
        # staying meaningful regardless of the pipeline's absolute calibration scale.
        high_blue_ratio = float(np.mean(blue_channel > (mean_val + 0.5 * std_val))) if std_val > 1e-6 else 0.0
        
        # Blue haze factor: ratio of blue mean to overall RGB mean
        rgb_mean = float(np.mean(image_rgb)) + 1e-6
        blue_dominance = mean_val / rgb_mean

        return {
            "blue_mean": round(mean_val, 3),
            "blue_std": round(std_val, 3),
            "blue_skewness": round(skew_val, 3),
            "blue_high_ratio": round(high_blue_ratio, 4),
            "blue_dominance": round(blue_dominance, 3),
        }

    def extract_edge_sharpness_features(self, image_rgb: np.ndarray) -> Dict[str, float]:
        """
        2. Edge Detection Sharpness Score:
        Aerosols and smog diffuse light, suppressing sharp edges and structural contrast.
        Computes variance of Laplacian and Sobel gradient energy.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Variance of Laplacian (classic sharpness / focus metric)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(laplacian.var())

        # Sobel Gradients in X and Y
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_mean = float(np.mean(grad_mag))
        sobel_std = float(np.std(grad_mag))

        # Edge pixel density using a data-driven ("auto-Canny") threshold pair instead of
        # fixed absolute levels. This used to call cv2.Canny(gray, 50, 150) with hardcoded
        # thresholds while computing (and discarding) an Otsu threshold right above it.
        # These satellite patches have a much lower local gradient range than 50-150
        # assumes, so the fixed thresholds never fired and edge_density was silently 0.0
        # for every sample. Deriving the thresholds from the image's own median intensity
        # keeps the feature responsive across both clear and hazy patches.
        median_val = float(np.median(gray))
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * median_val))
        upper = int(min(255, (1.0 + sigma) * median_val))
        if upper <= lower:
            upper = lower + 1
        canny_edges = cv2.Canny(gray, lower, upper)
        edge_density = float(np.mean(canny_edges > 0))

        return {
            "laplacian_var": round(laplacian_var, 3),
            "sobel_mean": round(sobel_mean, 3),
            "sobel_std": round(sobel_std, 3),
            "edge_density": round(edge_density, 4),
        }

    def extract_hsi_features(self, image_rgb: np.ndarray) -> Dict[str, float]:
        """
        3. Color Analysis in HSI / HSV Space:
        Smog heavily attenuates color saturation (converting vivid vegetation/urban surfaces
        into a low-saturation milky veil).
        """
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        # S channel (0-255 in OpenCV) -> normalize to 0-1
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0
        
        sat_mean = float(np.mean(sat))
        sat_std = float(np.std(sat))
        
        # Dark Channel Prior concept: in clean non-sky regions, at least one channel has very low intensity
        dark_channel = np.min(image_rgb, axis=2)
        dark_mean = float(np.mean(dark_channel))
        
        # Haze Index: High Value (brightness) + Low Saturation = High Smog
        haze_index = float(np.mean(val / (sat + 0.05)))

        return {
            "hsi_saturation_mean": round(sat_mean, 4),
            "hsi_saturation_std": round(sat_std, 4),
            "dark_channel_mean": round(dark_mean, 3),
            "haze_index": round(haze_index, 3),
        }

    def extract_fft_frequency_features(self, image_rgb: np.ndarray) -> Dict[str, float]:
        """
        4. Frequency-Domain 2D-FFT Analysis:
        Particulate matter dampens high-frequency spatial harmonics.
        Computes the ratio of high-frequency energy to total spectral energy.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape
        
        # 2D Fast Fourier Transform
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        energy_spectrum = magnitude_spectrum ** 2
        
        total_energy = float(np.sum(energy_spectrum)) + 1e-8
        
        # Create circular high-pass filter mask
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        radius = min(h, w) * self.high_freq_radius_ratio
        
        high_freq_mask = dist_from_center >= radius
        high_energy = float(np.sum(energy_spectrum[high_freq_mask]))
        
        high_energy_ratio = high_energy / total_energy
        low_energy_ratio = 1.0 - high_energy_ratio

        return {
            "fft_high_energy_ratio": round(float(high_energy_ratio), 5),
            "fft_low_energy_ratio": round(float(low_energy_ratio), 5),
            "fft_spectral_energy_log": round(float(np.log1p(total_energy)), 3),
        }

    def extract_supplementary_features(self, image_rgb: np.ndarray) -> Dict[str, float]:
        """
        5. Supplementary DIP Features:
        Three additional physically-grounded descriptors that capture smog signal
        orthogonal to the four primary DIP methods above.

        a) Gray Contrast Range:
           Smog compresses the scene's dynamic range — bright objects become dim
           and dark shadows are lifted toward a uniform gray veil. The difference
           between the 95th and 5th intensity percentile shrinks monotonically with
           aerosol optical depth in cloud-free satellite scenes (Koschmieder model).

        b) Blue-to-Red Ratio (Mie Scattering Proxy):
           Fine particulate matter (PM2.5, d < 2.5 µm) preferentially forward-scatters
           shorter wavelengths (Mie regime for d ~ λ). In a smoggy scene the blue
           channel is brightened relative to the red channel, so the B/R ratio rises
           with smog loading. This is a cross-channel ratio that the four single-channel
           DIP extractors above do not compute directly.

        c) Local Variance Mean (Spatial Texture Loss):
           Aerosol haze acts as a spatial low-pass filter: fine surface textures
           (crop rows, road markings, building edges) are blurred into homogeneous
           patches. The mean variance computed over non-overlapping 8×8 pixel blocks
           of the luminance channel captures this texture loss; clear scenes have
           high block variance, smoggy scenes have near-uniform low-variance blocks.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # a) Contrast range: p95 – p5 of gray intensity
        gray_contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))

        # b) Blue-to-Red ratio (Mie scattering proxy)
        red_mean  = float(np.mean(image_rgb[:, :, 0])) + 1e-5
        blue_mean = float(np.mean(image_rgb[:, :, 2])) + 1e-5
        blue_red_ratio = blue_mean / red_mean

        # c) Mean of 8×8 block variances (spatial texture loss)
        h, w = gray.shape
        block = 8
        block_vars = []
        for y0 in range(0, h - block + 1, block):
            for x0 in range(0, w - block + 1, block):
                tile = gray[y0:y0 + block, x0:x0 + block]
                block_vars.append(float(np.var(tile)))
        local_variance_mean = float(np.mean(block_vars)) if block_vars else 0.0

        return {
            "gray_contrast":       round(gray_contrast, 3),
            "blue_red_ratio":      round(blue_red_ratio, 4),
            "local_variance_mean": round(local_variance_mean, 3),
        }

    def _normalize_to_uint8(self, image_rgb: np.ndarray) -> np.ndarray:
        """Safely normalizes any input array (float, uint16, uint8) into a uint8 RGB image."""
        if image_rgb.dtype == np.uint8:
            out = image_rgb.copy()
        else:
            arr = image_rgb.astype(np.float32)
            max_v = float(np.nanmax(arr)) if arr.size > 0 else 1.0
            min_v = float(np.nanmin(arr)) if arr.size > 0 else 0.0
            
            if max_v <= 1.01:
                out = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            elif max_v <= 255.0 and min_v >= 0.0:
                out = np.clip(arr, 0, 255).astype(np.uint8)
            else:
                # 16-bit GeoTIFF DN values (e.g. 0..10000) or high dynamic range images
                scale = max_v - min_v if max_v > min_v else 1.0
                out = np.clip((arr - min_v) / scale * 255.0, 0, 255).astype(np.uint8)
                
        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
        elif out.shape[2] == 4:
            out = cv2.cvtColor(out, cv2.COLOR_RGBA2RGB)
        return out

    def compute_dip_optical_aod(self, features_or_image: Any) -> float:
        """
        Computes physical Koschmieder optical haze / DIP Aerosol Optical Depth (AOD)
        estimate from the image's classical DIP feature vector or raw RGB image array.
        
        Outputs estimated AOD magnitude in range [0.08, 1.25]:
          - Clear / Good:        AOD ~ 0.08 - 0.18
          - Moderate Haze:       AOD ~ 0.18 - 0.38
          - Unhealthy Smog:      AOD ~ 0.38 - 0.75
          - Hazardous Emergency: AOD ~ 0.75 - 1.25
        """
        if isinstance(features_or_image, dict):
            f = features_or_image
        else:
            f = self.extract_all_features(features_or_image)
            
        lap_score = np.clip(1.0 - (f.get("laplacian_var", 400.0) / 600.0), 0.0, 1.0)
        sat_score = np.clip(1.0 - (f.get("hsi_saturation_mean", 0.25) / 0.30), 0.0, 1.0)
        contrast_score = np.clip(1.0 - (f.get("gray_contrast", 50.0) / 70.0), 0.0, 1.0)
        texture_score = np.clip(1.0 - (f.get("local_variance_mean", 150.0) / 200.0), 0.0, 1.0)
        haze_score = np.clip((f.get("haze_index", 3.0) - 3.0) / 12.0, 0.0, 1.0)
        
        aod = 0.08 + 0.35 * lap_score + 0.25 * sat_score + 0.20 * contrast_score + 0.15 * texture_score + 0.20 * haze_score
        return round(float(np.clip(aod, 0.08, 1.25)), 3)

    def extract_all_features(self, image_rgb: np.ndarray) -> Dict[str, float]:
        """Runs the complete DIP pipeline and returns a unified tabular feature dictionary."""
        image_rgb = self._normalize_to_uint8(image_rgb)
            
        features = {}
        features.update(self.extract_blue_histogram_features(image_rgb))
        features.update(self.extract_edge_sharpness_features(image_rgb))
        features.update(self.extract_hsi_features(image_rgb))
        features.update(self.extract_fft_frequency_features(image_rgb))
        features.update(self.extract_supplementary_features(image_rgb))
        
        # Attach optical DIP AOD estimate
        features["dip_optical_aod"] = self.compute_dip_optical_aod(features)
        return features

    def generate_diagnostic_visualizations(self, image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generates 4 visual maps corresponding to the 4 DIP features:
        1. Blue Channel Heatmap
        2. Laplacian Edge Detection Map
        3. HSV Saturation Color Map
        4. 2D-FFT Log Magnitude Spectrum Map
        """
        image_rgb = self._normalize_to_uint8(image_rgb)
            
        # 1. Blue channel mapped to colormap
        blue = image_rgb[:, :, 2]
        blue_viz = cv2.applyColorMap(blue, cv2.COLORMAP_OCEAN)
        blue_viz = cv2.cvtColor(blue_viz, cv2.COLOR_BGR2RGB)
        
        # 2. Laplacian Edge Map
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_norm = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        lap_viz = cv2.applyColorMap(lap_norm, cv2.COLORMAP_INFERNO)
        lap_viz = cv2.cvtColor(lap_viz, cv2.COLOR_BGR2RGB)
        
        # 3. Saturation Map
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        sat_viz = cv2.applyColorMap(sat, cv2.COLORMAP_VIRIDIS)
        sat_viz = cv2.cvtColor(sat_viz, cv2.COLOR_BGR2RGB)
        
        # 4. FFT Magnitude Spectrum
        f = fft2(gray.astype(np.float32))
        fshift = fftshift(f)
        mag = np.log1p(np.abs(fshift))
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        fft_viz = cv2.applyColorMap(mag_norm, cv2.COLORMAP_MAGMA)
        fft_viz = cv2.cvtColor(fft_viz, cv2.COLOR_BGR2RGB)
        
        return {
            "blue_channel": blue_viz,
            "edge_laplacian": lap_viz,
            "saturation_map": sat_viz,
            "fft_spectrum": fft_viz,
        }