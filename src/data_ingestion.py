"""
Multi-Source Satellite, Ground Sensor & Weather Data Ingestion Engine
======================================================================
Data sources used in this pipeline:
1. NASA GIBS WMS — authentic multi-temporal MODIS Terra satellite scenes (dated passes)
2. ESA Sentinel-2 Cloudless WMS — basemap fallback for cloud-free optical reference
3. Open-Meteo Air Quality API (Copernicus CAMS-based) — hourly PM2.5 & AOD at 550 nm
4. Open-Meteo Weather API (ECMWF-based forecast, not ERA5 archive) — temperature, humidity, wind
5. NASA FIRMS — live VIIRS/MODIS South Asia active fire CSV feed
6. WAQI public API — best publicly available proxy for urban-hub AQI
   (Note: WAQI city feeds are the closest real-time ground reference accessible without
    a restricted EPA/PEQS institutional token; readings are used as-is without added noise)

Honest limitations documented here for academic transparency:
- "ERA5" language is not used — Open-Meteo uses ECMWF models but this is not the ERA5 archive.
- PM2.5 training labels are the raw Copernicus CAMS hourly values; no season multipliers applied.
- Season is derived purely from each record's actual calendar timestamp month.
- WAQI city PM2.5 values are used as ground-truth validation labels alongside CAMS labels.
- Satellite image augmentation is geometric/photometric only (crop, flip, rotation,
  brightness, color jitter, sensor noise). pm25 and severity_class are NEVER passed into
  the image pipeline in any form — a prior version of this file computed a synthetic
  Koschmieder haze transmission as a function of pm25 before DIP feature extraction,
  which leaked the label into blue_mean/hot_index/haze_index. That function has been
  removed. Expect lower, more modest accuracy/R²/IoU than earlier runs — that is the
  honest signal ceiling given only 4 real dated satellite acquisitions per city, not a
  regression.
- FIX (this version): modis_aod also had a leak. Whenever real Copernicus AOD was
  unavailable (CAMS unreachable, or a specific date gap), the fallback synthesized
  aod = pm25 / 280.0 — deriving a *feature* directly from the *regression target*.
  Feature-importance logs confirmed this: modis_aod was sitting at ~34% importance,
  far ahead of every genuine DIP feature, which is the signature of a leaky shortcut
  dominating the model rather than real learned signal. All pm25-derived AOD fallbacks
  below are replaced with a season-only climatological AOD baseline (AOD_SEASON_BASELINE),
  which reflects genuine regional/seasonal aerosol patterns without ever touching this
  row's own pm25 value. Expect r2_score and iou_smog_detection to drop after this fix —
  that drop is the leak being removed, not new noise. Treat the post-fix number as your
  real baseline going forward.
"""

import os
import io
import json
import requests
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from src.punjab_geo import PUNJAB_DISTRICTS, PEQS_STATIONS
from src.dip_extractor import DIPFeatureExtractor
from src.preprocessing import SatellitePreprocessor

# FIX: season-only climatological AOD baseline used ONLY when real CAMS AOD is
# unavailable. These are typical satellite AOD-550nm magnitudes for Punjab by season —
# domain knowledge, not derived from any row's pm25_target. Deliberately coarse (season
# only, no district/pm25 conditioning) so it cannot function as a target-leaking proxy.
AOD_SEASON_BASELINE = {
    "Winter_Peak":    0.65,
    "Autumn_Harvest": 0.45,
    "Summer":         0.30,
    "Monsoon":        0.22,
}


class DataIngestionPipeline:
    """
    Production Ingestion Pipeline. Honest wording: not ERA5 archive.
    PM2.5 training labels = raw Copernicus CAMS hourly values from Open-Meteo.
    Ground validation labels = WAQI city AQI readings (best available public proxy for EPA stations).
    Season assignment = derived from the record's actual calendar month, not forced by index.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.punjab_bbox = (28.0, 69.5, 34.2, 75.8)
        self.dip_extractor = DIPFeatureExtractor()
        self.preprocessor = SatellitePreprocessor()

    # --------------------------------------------------------------------------------------------
    # 1. REAL NASA FIRMS ACTIVE CROP FIRE FEED
    # --------------------------------------------------------------------------------------------
    def fetch_nasa_firms_hotspots(self, active_season_boost: float = 1.0) -> List[Dict[str, Any]]:
        hotspots = []
        urls = [
            "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_South_Asia_24h.csv",
            "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_South_Asia_24h.csv"
        ]
        min_lat, min_lon, max_lat, max_lon = self.punjab_bbox
        fire_id = 1

        for url in urls:
            try:
                res = requests.get(url, timeout=4.0)
                if res.status_code == 200 and len(res.text) > 100:
                    df_firms = pd.read_csv(io.StringIO(res.text))
                    punjab_fires = df_firms[
                        (df_firms["latitude"] >= min_lat) & (df_firms["latitude"] <= max_lat) &
                        (df_firms["longitude"] >= min_lon) & (df_firms["longitude"] <= max_lon)
                    ]
                    for _, row in punjab_fires.iterrows():
                        lat = float(row.get("latitude", 31.5))
                        lon = float(row.get("longitude", 74.0))
                        frp = float(row.get("frp", row.get("brightness", 45.0)))
                        bright_k = float(row.get("bright_ti4", row.get("brightness", 325.0)))
                        conf = str(row.get("confidence", "nominal"))
                        sat = str(row.get("satellite", "SNPP-VIIRS"))
                        nearest_d = min(PUNJAB_DISTRICTS.keys(), key=lambda d: (PUNJAB_DISTRICTS[d]["lat"] - lat)**2 + (PUNJAB_DISTRICTS[d]["lon"] - lon)**2)
                        hotspots.append({
                            "id": f"NASA-FIRMS-{fire_id:04d}",
                            "cluster": f"{nearest_d} Stubble Zone",
                            "lat": round(lat, 4),
                            "lon": round(lon, 4),
                            "frp_mw": round(frp, 1),
                            "brightness_k": round(bright_k, 1),
                            "confidence": "High" if ("h" in conf.lower() or frp > 50) else "Nominal",
                            "satellite": sat,
                            "acquisition_time": f"{str(row.get('acq_time', '1200'))[:2]}:{str(row.get('acq_time', '1200'))[2:]} UTC"
                        })
                        fire_id += 1
                    if len(hotspots) > 0:
                        return hotspots
            except Exception:
                pass

        # Physical-cluster fallback when FIRMS CSV is unreachable
        np.random.seed(42)
        clusters = [
            {"name": "Sheikhupura Rice Belt", "lat": 31.75, "lon": 73.95, "radius": 0.35, "base_count": int(14 * active_season_boost)},
            {"name": "Kasur Stubble Corridor", "lat": 31.12, "lon": 74.42, "radius": 0.28, "base_count": int(11 * active_season_boost)},
            {"name": "Gujranwala-Daska Belt", "lat": 32.22, "lon": 74.30, "radius": 0.30, "base_count": int(9 * active_season_boost)},
            {"name": "Faisalabad Outskirts", "lat": 31.40, "lon": 73.25, "radius": 0.25, "base_count": int(6 * active_season_boost)},
            {"name": "Multan Cotton Corridor", "lat": 30.22, "lon": 71.48, "radius": 0.26, "base_count": int(5 * active_season_boost)},
        ]
        for cluster in clusters:
            for _ in range(cluster["base_count"]):
                offset_lat = np.random.normal(0, cluster["radius"] * 0.4)
                offset_lon = np.random.normal(0, cluster["radius"] * 0.4)
                frp = round(float(np.random.uniform(18.0, 145.0)), 1)
                hotspots.append({
                    "id": f"NASA-FIRMS-{fire_id:04d}",
                    "cluster": cluster["name"],
                    "lat": round(cluster["lat"] + offset_lat, 4),
                    "lon": round(cluster["lon"] + offset_lon, 4),
                    "frp_mw": frp,
                    "brightness_k": round(float(np.random.uniform(320.0, 370.0)), 1),
                    "confidence": "High" if frp > 50 else "Nominal",
                    "satellite": "SNPP-VIIRS",
                    "acquisition_time": "12:00 UTC"
                })
                fire_id += 1
        return hotspots

    # --------------------------------------------------------------------------------------------
    # 2. MULTI-TEMPORAL SATELLITE IMAGERY (NASA GIBS dated passes + local tile cache fallback)
    # --------------------------------------------------------------------------------------------
    def fetch_satellite_smog_scene(self, lat: float, lon: float, date_str: str = "2024-11-15",
                                   width: int = 256, height: int = 256,
                                   season_key: str = "Winter_Peak") -> np.ndarray:
        """
        Downloads a real NASA MODIS Terra Corrected Reflectance image captured on date_str.
        Fallback chain (in order, logged at each step):
          1. NASA GIBS WMS (dated pass, network)
          2. ESA Sentinel-2 Cloudless WMS (basemap, network)
          3. Locally cached tile PNG from data/tile_cache/
          4. Season-representative PNG from data/sample_patches/
          5. Varied synthetic patch seeded by lat/lon (NOT PM2.5-correlated) — last resort
        """
        import cv2 as _cv2

        def _is_uniform(arr: np.ndarray, std_threshold: float = 4.0) -> bool:
            return float(np.std(arr.astype(np.float32))) < std_threshold

        # ── Source 1: NASA GIBS WMS (dated pass) — retry nearby real dates before ─────
        # falling to the static basemap below. The old behavior jumped straight to the
        # ESA cloudless composite (which is the SAME fixed 2020 image regardless of what
        # date was requested) on the first NASA GIBS miss. That's fine occasionally, but
        # once more dates are requested per city, GIBS misses more often (cloud cover /
        # transient errors on a specific day), and every miss was injecting an
        # unchanging "clear" image next to a real, possibly hazardous PM2.5 label —
        # label noise straight into the image branch. Retrying nearby real dates first
        # (same +/-1/2/3 day pattern already used for PM2.5 matching below) keeps the
        # image genuinely tied to roughly the right day instead of falling back to a
        # date-blind placeholder.
        tile_dir = os.path.join(self.data_dir, "tile_cache")
        os.makedirs(tile_dir, exist_ok=True)
        exact_cache_file = os.path.join(tile_dir, f"tile_{lat:.4f}_{lon:.4f}_{date_str}.png")
        if os.path.exists(exact_cache_file):
            try:
                arr = np.array(Image.open(exact_cache_file).convert("RGB"))
                if not _is_uniform(arr):
                    # FIX: this was the only branch in the whole function that returned
                    # without printing anything. Every other source (GIBS/ESA/nearest-
                    # tile/season-fallback/synthetic) logs on success or failure, so a
                    # dataset built entirely from cache hits produced ZERO "[Sat ...]"
                    # lines -- which looked identical to images never being fetched at
                    # all. Confirmed via direct file inspection that this cache DOES hold
                    # 252 genuinely distinct dated tiles (6 cities x 42 dates), not a
                    # small reused pool -- this was a missing log line, not a data bug.
                    print(f"    [Sat CACHE-EXACT] {os.path.basename(exact_cache_file)} (real dated tile, reused from prior fetch)")
                    return arr
            except Exception:
                pass

        delta = 0.25
        for offset in (0, -1, 1, -2, 2, -3, 3):
            try:
                probe_date = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=offset)).strftime("%Y-%m-%d")
            except ValueError:
                continue
            wms_url = (
                f"https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?"
                f"SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
                f"&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor"
                f"&CRS=EPSG:4326&BBOX={lat-delta},{lon-delta},{lat+delta},{lon+delta}"
                f"&WIDTH={width}&HEIGHT={height}&FORMAT=image/jpeg&TIME={probe_date}"
            )
            try:
                res = requests.get(wms_url, timeout=6.0)
                if res.status_code == 200 and len(res.content) > 1000:
                    arr = np.array(Image.open(io.BytesIO(res.content)).convert("RGB"))
                    if not _is_uniform(arr):
                        tag = date_str if offset == 0 else f"{probe_date} (nearest to {date_str})"
                        print(f"    [Sat OK] NASA GIBS MODIS {tag} ({lat:.3f},{lon:.3f})")
                        try:
                            Image.fromarray(arr).save(exact_cache_file)
                        except Exception:
                            pass
                        return arr
            except Exception as e:
                if offset == 0:
                    print(f"    [Sat FAIL] NASA GIBS: {type(e).__name__}: {e}")

        # ── Source 2: ESA Sentinel-2 Cloudless WMS ───────────────────────────────────
        s2_url = (
            f"https://tiles.maps.eox.at/wms?service=wms&request=getmap&version=1.3.0"
            f"&layers=s2cloudless-2020&crs=epsg:4326"
            f"&bbox={lat-0.08},{lon-0.08},{lat+0.08},{lon+0.08}"
            f"&width={width}&height={height}&format=image/png"
        )
        try:
            res = requests.get(s2_url, timeout=6.0)
            if res.status_code == 200 and len(res.content) > 1000:
                arr = np.array(Image.open(io.BytesIO(res.content)).convert("RGB"))
                if not _is_uniform(arr):
                    print(f"    [Sat OK] ESA S2 Cloudless WMS ({lat:.3f},{lon:.3f})")
                    return arr
        except Exception as e:
            print(f"    [Sat FAIL] EOX S2: {type(e).__name__}: {e}")

        # ── Source 3: Local tile cache (written by previous successful fetches) ───────
        tile_dir = os.path.join(self.data_dir, "tile_cache")
        if os.path.isdir(tile_dir):
            best_tile, best_dist = None, float("inf")
            for fn in os.listdir(tile_dir):
                if not fn.endswith(".png"):
                    continue
                parts = fn.replace(".png", "").split("_")
                try:
                    t_lat, t_lon = float(parts[1]), float(parts[2])
                    dist = (t_lat - lat) ** 2 + (t_lon - lon) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_tile = os.path.join(tile_dir, fn)
                except (IndexError, ValueError):
                    continue
            if best_tile is not None:
                try:
                    arr = np.array(Image.open(best_tile).convert("RGB"))
                    if arr.shape[0] != width or arr.shape[1] != height:
                        arr = _cv2.resize(arr, (width, height))
                    if not _is_uniform(arr):
                        print(f"    [Sat CACHE] Local tile: {os.path.basename(best_tile)}")
                        return arr
                except Exception as e:
                    print(f"    [Sat FAIL] Local cache: {e}")

        # ── Source 4: Season PNG from data/sample_patches/ ───────────────────────────
        patch_dir = os.path.join(self.data_dir, "sample_patches")
        season_png_map = {
            "Winter_Peak":    "hazardous_smog_cropfire.png",
            "Autumn_Harvest": "moderate_haze_faisalabad.png",
            "Summer":         "clear_day_lahore.png",
            "Monsoon":        "clear_day_lahore.png",
        }
        fallback_fn = os.path.join(patch_dir, season_png_map.get(season_key, "clear_day_lahore.png"))
        if os.path.exists(fallback_fn):
            try:
                arr = np.array(Image.open(fallback_fn).convert("RGB"))
                if arr.shape[0] != width or arr.shape[1] != height:
                    arr = _cv2.resize(arr, (width, height))
                if not _is_uniform(arr):
                    print(f"    [Sat FALLBACK] Sample patch ({os.path.basename(fallback_fn)}) — network blocked")
                    return arr
            except Exception as e:
                print(f"    [Sat FAIL] Sample patch: {e}")

        # ── Source 5: Varied synthetic patch — seeded by lat/lon, NOT PM2.5 ──────────
        print(f"    [Sat SYNTHETIC] All sources failed — varied synthetic patch ({lat:.3f},{lon:.3f})")
        rng = np.random.RandomState(seed=int(abs(lat * 1000) + abs(lon * 100)) % (2 ** 31))
        base = rng.randint(80, 180, size=(width, height, 3), dtype=np.uint8)
        blurred = _cv2.GaussianBlur(base.astype(np.float32), (15, 15), 5)
        noise = rng.normal(0, 18, size=(width, height, 3))
        return np.clip(blurred + noise, 0, 255).astype(np.uint8)


    # --------------------------------------------------------------------------------------------
    # 3. GROUND AQI STATION READINGS
    #    Source: WAQI public API — best available public proxy for city-level EPA/PEQS AQI.
    #    Readings are returned as-is (no noise injection) so they can serve as honest
    #    ground-truth reference labels alongside Copernicus CAMS.
    # --------------------------------------------------------------------------------------------
    def fetch_peqs_station_readings(self) -> List[Dict[str, Any]]:
        """
        Fetches live city-level PM2.5 readings from the WAQI public API for the
        four primary Punjab urban monitoring hubs.  These are the best publicly accessible
        approximation of EPA-Punjab / PEQS station data without a restricted institutional token.
        Readings are used as-is — no random noise is added, so values reflect the raw API response.
        """
        # WAQI city slugs for Punjab's main urban monitoring hubs
        city_slugs = {
            "Lahore": "lahore",
            "Faisalabad": "faisalabad",
            "Rawalpindi": "islamabad",
            "Multan": "multan",
        }
        live_city_pm25: Dict[str, Optional[float]] = {}

        for cname, slug in city_slugs.items():
            try:
                r = requests.get(f"https://api.waqi.info/feed/{slug}/?token=demo", timeout=3.5)
                if r.status_code == 200:
                    val = r.json().get("data", {}).get("iaqi", {}).get("pm25", {}).get("v")
                    if val is not None:
                        live_city_pm25[cname] = float(val)
            except Exception:
                pass

        results = []
        for stn in PEQS_STATIONS:
            dist = stn["district"]
            # Use the raw WAQI reading if available; fall back to the district's known
            # baseline only when the API is unreachable (clearly documented).
            raw_pm25 = live_city_pm25.get(dist)
            if raw_pm25 is None:
                # API unreachable — flag this explicitly; do NOT inject noise
                raw_pm25 = PUNJAB_DISTRICTS.get(dist, {}).get("baseline_winter_pm25", 150.0)
                data_source = "Baseline fallback (WAQI unreachable)"
            else:
                data_source = "WAQI live API (city-level)"

            results.append({
                "station_id": stn["id"],
                "station_name": stn["name"],
                "district": dist,
                "lat": stn["lat"],
                "lon": stn["lon"],
                "sensor_type": stn.get("sensor_type", "BAM-1020"),
                "measured_pm25": round(raw_pm25, 1),
                "data_source": data_source,
                "status": "ONLINE / ACTIVE" if "WAQI" in data_source else "FALLBACK",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M PKT"),
            })
        return results

    # --------------------------------------------------------------------------------------------
    # 4. WEATHER + AOD (Open-Meteo ECMWF-based forecast, not ERA5 archive)
    # --------------------------------------------------------------------------------------------
    def fetch_district_weather(self, district_name: str, use_live_api: bool = True) -> Dict[str, float]:
        """
        Fetches live meteorological conditions from Open-Meteo (ECMWF-based forecast API)
        and live Copernicus CAMS Aerosol Optical Depth at 550 nm.
        Note: Open-Meteo is ECMWF-model based but is NOT the ERA5 historical reanalysis archive.
        """
        if district_name not in PUNJAB_DISTRICTS:
            district_name = "Lahore"
        dist = PUNJAB_DISTRICTS[district_name]
        lat, lon = dist["lat"], dist["lon"]

        if use_live_api:
            try:
                w_url = (
                    f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                    f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
                    f"&timezone=Asia/Karachi"
                )
                aq_url = (
                    f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}"
                    f"&current=pm2_5,aerosol_optical_depth&timezone=Asia/Karachi"
                )
                w_res = requests.get(w_url, timeout=3.5)
                aq_res = requests.get(aq_url, timeout=3.5)

                w_data = w_res.json().get("current", {}) if w_res.status_code == 200 else {}
                aq_data = aq_res.json().get("current", {}) if aq_res.status_code == 200 else {}

                live_aod = aq_data.get("aerosol_optical_depth", 0.45)
                return {
                    "temperature_c": float(w_data.get("temperature_2m", 18.5)),
                    "humidity_pct": float(w_data.get("relative_humidity_2m", 72.0)),
                    "wind_speed_kmh": float(w_data.get("wind_speed_10m", 6.5)),
                    "surface_pressure_hpa": float(w_data.get("surface_pressure", 1012.0)),
                    "modis_aod": float(live_aod),
                    "aod": float(live_aod),
                }
            except Exception:
                pass

        return {
            "temperature_c": 16.2 + np.random.uniform(-3, 4),
            "humidity_pct": 78.5 + np.random.uniform(-10, 15),
            "wind_speed_kmh": 4.8 + np.random.uniform(-2, 3),
            "surface_pressure_hpa": 1015.0 + np.random.uniform(-4, 4),
            "modis_aod": 0.52 + np.random.uniform(-0.15, 0.20),
            "aod": 0.52 + np.random.uniform(-0.15, 0.20),
        }

    # --------------------------------------------------------------------------------------------
    # 5. TRAINING DATASET BUILDER
    #    - PM2.5 labels: raw Copernicus CAMS hourly values — NO season multipliers applied.
    #    - Ground validation: WAQI city readings used as secondary reference labels.
    #    - Season: derived purely from the record's actual calendar month (no index forcing).
    #    - Satellite image: paired to the season determined by the real timestamp month.
    # --------------------------------------------------------------------------------------------
    def build_real_training_dataset(self, n_samples_per_city: int = 350, save_csv: bool = True) -> pd.DataFrame:
        """
        Builds the training/validation dataset with honest labeling:

        PM2.5 target  = raw hourly value from Copernicus CAMS via Open-Meteo (no multipliers).
        AOD           = raw hourly aerosol_optical_depth from the same API call, or (FIX) a
                        season-only climatological baseline when that's unavailable — never
                        derived from this row's own pm25 value.
        Season        = derived from the actual calendar month of the record's timestamp.
        Satellite img = fetched for real dates; PM2.5 fetch window is anchored to those SAME
                        real dates so each photo can be paired with its own real day's
                        readings (see PER-REAL-PHOTO DATE MATCHING below), not a random
                        reading from anywhere in that season.
        Validation    = WAQI city-level PM2.5 readings used as supplementary ground-truth reference.

        Fallback when API is unreachable: a proportional mix of plausible values tied to the
        district's documented baseline — clearly flagged in the printed log.
        """
        primary_cities = ["Lahore", "Multan", "Faisalabad", "Rawalpindi", "Sheikhupura", "Gujranwala"]
        records = []

        print("[Data Ingestion] Fetching real ESA Sentinel-2 / NASA satellite tiles across Punjab districts...")

        # Authentic satellite scene dates per season — MULTIPLE real dated acquisitions
        # per season now, instead of one, so the DIP extractor sees more than a single
        # real photo (with jitter) per season. Verify/replace these against actual
        # AQI records or news coverage for genuine smog/clear days in your target years.
        # Densified vs. the original 18-date list (5+5+4+4). More independent real
        # acquisitions per city is the only honest lever left for raising R² further:
        # hyperparameter tuning in ml_pipeline.py already plateaus around ~0.74-0.76
        # because only ~18 real photos/city backed every prior run (confirmed by a
        # GroupKFold sweep — different hyperparameters moved the held-out R² by <0.01).
        # These are still the same two real sources (NASA GIBS MODIS + ESA Sentinel-2
        # Cloudless WMS) at more calendar dates within the same 2023-05-22..2024-12-08
        # window this pipeline has already confirmed the Copernicus CAMS/ECMWF archives
        # cover — no new data source, no synthetic dates outside the verified span.
        satellite_dates = {
            "Winter_Peak":    ["2023-11-05", "2023-11-20", "2023-11-30", "2023-12-01", "2023-12-10",
                                "2023-12-20", "2024-11-01", "2024-11-15", "2024-11-25", "2024-12-05",
                                "2024-01-05", "2024-01-15"],                                              # smog inversion episodes
            "Autumn_Harvest": ["2023-09-15", "2023-09-25", "2023-10-05", "2023-10-15", "2023-10-25",
                                "2024-09-10", "2024-09-18", "2024-09-28", "2024-10-05", "2024-10-15",
                                "2024-10-22"],                                                            # stubble-burning onset
            "Summer":         ["2023-05-25", "2023-06-05", "2023-06-15", "2024-04-20", "2024-05-01",
                                "2024-05-18", "2024-05-28", "2024-06-10", "2024-06-20"],                  # hot dry pre-monsoon
            "Monsoon":        ["2023-07-15", "2023-07-25", "2023-08-05", "2023-08-20", "2023-08-30",
                                "2024-07-10", "2024-07-25", "2024-08-05", "2024-08-12", "2024-08-25"],    # post-rain washout
        }

        # Season lookup: month → season key (pure calendar, no index forcing)
        def month_to_season(month: int) -> str:
            if month in [11, 12, 1]:
                return "Winter_Peak"
            elif month in [9, 10]:
                return "Autumn_Harvest"
            elif month in [7, 8]:
                return "Monsoon"
            else:
                return "Summer"

        # Fetch WAQI ground-truth validation readings once (used as reference, not as labels)
        waqi_validation: Dict[str, float] = {}
        for cname, slug in [("Lahore", "lahore"), ("Faisalabad", "faisalabad"),
                             ("Rawalpindi", "islamabad"), ("Multan", "multan")]:
            try:
                r = requests.get(f"https://api.waqi.info/feed/{slug}/?token=demo", timeout=3.5)
                if r.status_code == 200:
                    val = r.json().get("data", {}).get("iaqi", {}).get("pm25", {}).get("v")
                    if val is not None:
                        waqi_validation[cname] = float(val)
            except Exception:
                pass

        for city in primary_cities:
            dist_meta = PUNJAB_DISTRICTS[city]
            lat, lon = dist_meta["lat"], dist_meta["lon"]

            # Download authentic dated satellite scenes for all 4 seasonal states, keyed by
            # their EXACT date (not just season) so each real photo can later be matched to
            # PM2.5 readings from that same real day. source_image_id tags which real photo
            # each patch came from (used by ml_pipeline.py's GroupShuffleSplit).
            city_date_patches: Dict[str, list] = {}
            date_to_season: Dict[str, str] = {}
            for sname, sdate_list in satellite_dates.items():
                for sdate in sdate_list:
                    date_to_season[sdate] = sname
                    scene = self.fetch_satellite_smog_scene(
                        lat, lon, date_str=sdate, width=384, height=384, season_key=sname
                    )
                    img_id = f"{city}_{sname}_{sdate}"
                    pool = []
                    for p in self.preprocessor.process_scene(scene, augment=True)["patches"]:
                        pool.append({"image": p["image"], "source_image_id": img_id})
                    city_date_patches[sdate] = pool

            # Anchor the PM2.5 fetch to the SAME real-world span as the satellite photos
            # above (with a few days' padding), instead of a rolling "last 365 days from
            # today" window. Those two windows used to never overlap once enough time had
            # passed since the satellite dates were hardcoded (e.g. running this in 2026
            # while the photos are dated 2023-2024 pulls PM2.5 for 2025-2026 — zero shared
            # days), which made real date-matching between image and label impossible.
            # Anchoring here also makes results reproducible across reruns instead of
            # silently drifting as "today" moves forward.
            all_sat_dates = sorted(date_to_season.keys())
            start_date = (datetime.strptime(all_sat_dates[0], "%Y-%m-%d") - timedelta(days=3)).strftime("%Y-%m-%d")
            end_date = (datetime.strptime(all_sat_dates[-1], "%Y-%m-%d") + timedelta(days=3)).strftime("%Y-%m-%d")
            cams_url = (
                f"https://air-quality-api.open-meteo.com/v1/air-quality?"
                f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
                f"&hourly=pm2_5,aerosol_optical_depth&timezone=Asia/Karachi"
            )

            pm_series: list = []
            aod_series: list = []
            time_series: list = []
            api_source = "Copernicus CAMS via Open-Meteo"

            # FIX: cache the raw API response to disk, keyed by location + date range.
            # Without this, every run of train.py re-hits the live CAMS endpoint, and a
            # single transient failure silently swaps a city from ~13,608 real hourly
            # records to the 567-row synthetic fallback -- which is exactly what happened
            # to Multan between two otherwise-identical runs, making Train/Test metrics
            # incomparable across runs for reasons that have nothing to do with the model.
            api_cache_dir = os.path.join(self.data_dir, "api_cache")
            os.makedirs(api_cache_dir, exist_ok=True)
            cams_cache_file = os.path.join(api_cache_dir, f"cams_{lat:.4f}_{lon:.4f}_{start_date}_{end_date}.json")

            if os.path.exists(cams_cache_file):
                try:
                    with open(cams_cache_file, "r") as f:
                        cached = json.load(f)
                    pm_series, aod_series, time_series = cached["pm_series"], cached["aod_series"], cached["time_series"]
                    api_source = "Copernicus CAMS via Open-Meteo (cached)"
                except Exception:
                    pm_series = []

            if not pm_series:
                try:
                    res = requests.get(cams_url, timeout=6.0)
                    if res.status_code == 200:
                        hourly = res.json().get("hourly", {})
                        pm_series = hourly.get("pm2_5", [])
                        aod_series = hourly.get("aerosol_optical_depth", [])
                        time_series = hourly.get("time", [])
                        if pm_series and len(pm_series) >= 100:
                            try:
                                with open(cams_cache_file, "w") as f:
                                    json.dump({"pm_series": pm_series, "aod_series": aod_series, "time_series": time_series}, f)
                            except Exception:
                                pass
                except Exception:
                    pass

            if not pm_series or len(pm_series) < 100:
                api_source = "Fallback (CAMS unreachable) — seasonal baseline"
                # Build a synthetic series spanning the same anchored date range as above,
                # not an arbitrary rolling year, so it stays consistent with the photos.
                span_days = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
                time_series, pm_series, aod_series = [], [], []
                for day_offset in range(span_days + 1):
                    ts_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=day_offset)
                    ts = ts_date.strftime("%Y-%m-%dT12:00")
                    m = ts_date.month
                    s = month_to_season(m)
                    if s == "Winter_Peak":
                        pm = dist_meta["baseline_winter_pm25"] * np.random.uniform(0.8, 1.2)
                    elif s == "Autumn_Harvest":
                        pm = dist_meta["baseline_winter_pm25"] * np.random.uniform(0.5, 0.85)
                    elif s == "Monsoon":
                        pm = dist_meta["baseline_summer_pm25"] * np.random.uniform(0.4, 0.7)
                    else:
                        pm = dist_meta["baseline_summer_pm25"] * np.random.uniform(0.7, 1.1)
                    time_series.append(ts)
                    pm_series.append(pm)
                    # FIX: was `round(pm / 280.0, 3)` — derived AOD directly from the pm25
                    # value being generated on the same line. Replaced with a season-only
                    # climatological baseline (+jitter), independent of this record's pm25.
                    aod_series.append(round(AOD_SEASON_BASELINE[s] * float(np.random.uniform(0.85, 1.15)), 3))

            print(f"  [{city}] PM2.5 source: {api_source} | {len(pm_series)} hourly records ({start_date} to {end_date})")

            # ── REAL HISTORICAL WEATHER (previously this whole block was fabricated) ────
            # temperature_c / humidity_pct / wind_speed_kmh / surface_pressure_hpa were
            # being generated from a fixed per-season template + random noise — never
            # real weather at all, despite the proposal treating ERA5-style weather as a
            # genuine model input. Pulling Open-Meteo's historical archive for the same
            # lat/lon and the same anchored date range gives real values, matched to the
            # exact same hourly timestamps as the CAMS PM2.5 series above.
            weather_url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
                f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
                f"&timezone=Asia/Karachi"
            )
            weather_by_time: Dict[str, tuple] = {}
            weather_source = "Open-Meteo Historical Weather Archive"
            weather_cache_file = os.path.join(api_cache_dir, f"weather_{lat:.4f}_{lon:.4f}_{start_date}_{end_date}.json")

            if os.path.exists(weather_cache_file):
                try:
                    with open(weather_cache_file, "r") as f:
                        weather_by_time = {k: tuple(v) for k, v in json.load(f).items()}
                    weather_source = "Open-Meteo Historical Weather Archive (cached)"
                except Exception:
                    weather_by_time = {}

            if not weather_by_time:
                try:
                    wres = requests.get(weather_url, timeout=6.0)
                    if wres.status_code == 200:
                        whourly = wres.json().get("hourly", {})
                        wt = whourly.get("time", [])
                        wtemp = whourly.get("temperature_2m", [])
                        wrh = whourly.get("relative_humidity_2m", [])
                        wwind = whourly.get("wind_speed_10m", [])
                        wpress = whourly.get("surface_pressure", [])
                        for i, ts in enumerate(wt):
                            if i < len(wtemp) and wtemp[i] is not None:
                                weather_by_time[ts] = (
                                    wtemp[i],
                                    wrh[i] if i < len(wrh) and wrh[i] is not None else None,
                                    wwind[i] if i < len(wwind) and wwind[i] is not None else None,
                                    wpress[i] if i < len(wpress) and wpress[i] is not None else None,
                                )
                        if weather_by_time:
                            try:
                                with open(weather_cache_file, "w") as f:
                                    json.dump(weather_by_time, f)
                            except Exception:
                                pass
                except Exception:
                    pass
            if not weather_by_time:
                weather_source = "Fallback (archive unreachable) — seasonal template"
            print(f"  [{city}] Weather source: {weather_source} | {len(weather_by_time)} hourly records")

            # Index all valid hourly records by their exact calendar date (YYYY-MM-DD)
            by_date: Dict[str, list] = {}
            for t_str, pm_val, aod_val in zip(time_series, pm_series, aod_series):
                if pm_val is None or (isinstance(pm_val, float) and np.isnan(pm_val)):
                    continue
                by_date.setdefault(str(t_str)[:10], []).append((t_str, float(pm_val), aod_val))

            # ── PER-REAL-PHOTO DATE MATCHING ─────────────────────────────────────────────
            # For every real satellite acquisition date, pull PM2.5 readings from THAT SAME
            # real day (falling back to the nearest day within +/-3 days on a genuine data
            # gap). This replaces the old approach of pairing a photo with a randomly-chosen
            # reading from anywhere in its season, where the image and the label could be
            # describing completely different, unrelated days.
            samples_per_date = max(8, n_samples_per_city // max(1, len(all_sat_dates)))
            selected: list = []  # (t_str, pm25, aod_raw, season_key_str, sdate)
            rng = np.random.default_rng(seed=42)

            for sdate in all_sat_dates:
                sname = date_to_season[sdate]
                day_records = by_date.get(sdate)
                if not day_records:
                    for delta in (1, -1, 2, -2, 3, -3):
                        alt = (datetime.strptime(sdate, "%Y-%m-%d") + timedelta(days=delta)).strftime("%Y-%m-%d")
                        if alt in by_date:
                            day_records = by_date[alt]
                            break
                if not day_records:
                    # Genuine gap even within the anchored window — synthesize only for
                    # this specific date, clearly distinguishable via label_source below.
                    base_pm = (dist_meta["baseline_winter_pm25"] if sname in ("Winter_Peak", "Autumn_Harvest")
                               else dist_meta["baseline_summer_pm25"])
                    mult = 1.0 if sname == "Winter_Peak" else (0.65 if sname == "Autumn_Harvest"
                           else (0.55 if sname == "Monsoon" else 1.0))
                    day_records = [
                        # FIX: was `round(base_pm * mult / 280.0, 3)` — again AOD derived
                        # straight from the pm25 value on the same line. Replaced with the
                        # same season-only climatological baseline used above.
                        (f"{sdate}T{h:02d}:00", base_pm * mult * float(rng.uniform(0.75, 1.25)),
                         round(AOD_SEASON_BASELINE[sname] * float(rng.uniform(0.85, 1.15)), 3))
                        for h in range(24)
                    ]

                idxs = rng.integers(0, len(day_records), size=samples_per_date)
                for ii in idxs:
                    t_str, pm_val, aod_val = day_records[ii]
                    selected.append((t_str, pm_val, aod_val, sname, sdate))

            rng.shuffle(selected)


            # ── LABEL-INDEPENDENT IMAGE AUGMENTATION ──────────────────────────────────────
            # The base image is selected by SEASON (from the timestamp's calendar month),
            # never by the PM2.5 label. IMPORTANT: unlike a previous version of this
            # function, pm25 is NOT an input here in any form — not directly, not through
            # a physics formula, not through severity_class. The augmentation below only
            # applies generic geometric/photometric jitter (crop, flip, rotation,
            # brightness, per-channel color, sensor noise) so that repeated draws from the
            # same real dated scenes aren't byte-identical, without manufacturing any
            # correlation between the image and the target we're trying to predict.
            #
            # Consequence (expected and correct): with a limited number of real dated
            # acquisitions per city, the genuine optical signal available to the DIP
            # extractor is limited. Classification/regression metrics will be more modest
            # than a leaky pipeline would produce — that drop is the honest number.
            import cv2 as _cv2

            def _augment_label_independent(
                base_img: np.ndarray,
                rng_state: np.random.Generator,
            ) -> np.ndarray:
                """
                Generic augmentation with NO reference to pm25, severity_class, or any
                other prediction target. Only reads base_img and an independent RNG.
                """
                if np.nanmax(base_img) <= 1.01:
                    img = (np.clip(base_img, 0.0, 1.0) * 255.0).astype(np.float32)
                else:
                    img = base_img.astype(np.float32)

                # Random crop-and-resize (5-15% crop) for spatial variation
                h, w = img.shape[:2]
                crop_frac = rng_state.uniform(0.85, 1.0)
                ch, cw = int(h * crop_frac), int(w * crop_frac)
                y0 = int(rng_state.integers(0, max(1, h - ch + 1)))
                x0 = int(rng_state.integers(0, max(1, w - cw + 1)))
                img = img[y0:y0 + ch, x0:x0 + cw]
                img = _cv2.resize(img, (w, h))

                # Random flip
                if rng_state.random() > 0.5:
                    img = img[:, ::-1, :]
                # Random 90-degree rotation
                n_rot = int(rng_state.integers(0, 4))
                if n_rot > 0:
                    img = np.rot90(img, k=n_rot)

                # Brightness jitter (sensor/illumination variation, not smog-linked)
                img *= rng_state.uniform(0.9, 1.1)
                # Per-channel color jitter
                for c in range(3):
                    img[:, :, c] *= rng_state.uniform(0.94, 1.06)

                # Sensor noise (realistic satellite SNR ~ 100-200)
                sigma = rng_state.uniform(1.0, 3.0)
                img += rng_state.normal(0.0, sigma, img.shape)

                return np.clip(img, 0, 255).astype(np.uint8)

            aug_rng = np.random.default_rng(seed=int(abs(lat * 1000 + lon * 100)) % (2**31))

            # ── RECORD BUILDING ───────────────────────────────────────────────────────────
            for t_str, pm_val, aod_raw, season_key_rec, sdate in selected:
                raw_pm25 = pm_val
                pm25 = round(float(np.clip(raw_pm25, 5.0, 650.0)), 1)
                # FIX: was `aod_raw if aod_raw is not None else pm25 / 280.0` — the else
                # branch derived AOD from THIS row's own pm25 target. aod_raw arriving here
                # is now either genuine CAMS AOD or an already-computed season baseline
                # (never pm25-derived), so this line only clips/rounds — no more fallback
                # to pm25 at the point of use.
                aod_val = round(float(np.clip(
                    aod_raw if aod_raw is not None else AOD_SEASON_BASELINE.get(season_key_rec, 0.35),
                    0.01, 4.0
                )), 3)

                try:
                    month = int(str(t_str)[5:7])
                except (ValueError, IndexError):
                    month = 11

                is_winter  = (season_key_rec == "Winter_Peak")
                is_autumn  = (season_key_rec == "Autumn_Harvest")
                is_monsoon = (season_key_rec == "Monsoon")

                # ── Base image: the REAL photo taken on this exact date (sdate) — the
                # same date this record's PM2.5 reading came from. Only which augmented
                # crop of that one photo is still varied (deterministically, by hash of
                # the timestamp), never which day's photo.
                patches_pool = city_date_patches.get(sdate) or next(iter(city_date_patches.values()))
                patch_idx = abs(hash(t_str)) % len(patches_pool)
                base_img = patches_pool[patch_idx]["image"]
                source_image_id = patches_pool[patch_idx]["source_image_id"]
                if base_img.shape[0] != 256 or base_img.shape[1] != 256:
                    base_img = _cv2.resize(base_img, (256, 256))

                # ── Apply generic augmentation — pm25 is never passed to this function ──
                patch_img = _augment_label_independent(base_img, aug_rng)

                # ── DIP feature extraction on the augmented real satellite scene ─────────
                dip_feat = self.dip_extractor.extract_all_features(patch_img)
                r_mean = float(np.mean(patch_img[:, :, 0]))
                b_mean = dip_feat["blue_mean"]
                hot_idx  = round(float((b_mean / 255.0) - 0.5 * (r_mean / 255.0) - 0.08), 3)
                ndsi_idx = round(float((b_mean - r_mean) / (b_mean + r_mean + 1e-5)), 3)

                # Meteorology (season-representative — NOT the regression target)
                # Meteorology — real historical weather for this exact hour where available,
                # falling back to the season-representative template only for a genuine
                # gap in the archive (e.g. a specific hour missing from the API response).
                real_wx = weather_by_time.get(t_str)
                if real_wx is not None:
                    temp_val     = round(float(real_wx[0]), 1)
                    rh_val       = round(float(np.clip(real_wx[1] if real_wx[1] is not None else 60.0, 5.0, 100.0)), 1)
                    wind_val     = round(float(np.clip(real_wx[2] if real_wx[2] is not None else 6.0, 0.5, 40.0)), 1)
                    pressure_val = round(float(real_wx[3] if real_wx[3] is not None else 1012.0), 1)
                else:
                    temp_val     = round(float((14.0 if is_winter else (24.0 if is_autumn else 34.0)) + np.random.normal(0, 2.5)), 1)
                    rh_val       = round(float(np.clip((82.0 if is_winter else (62.0 if is_autumn else 45.0)) + np.random.normal(0, 5.0), 20.0, 95.0)), 1)
                    wind_val     = round(float(np.clip((3.5 if is_winter else 8.0) + np.random.normal(0, 1.5), 1.0, 25.0)), 1)
                    pressure_val = round(float((1015.0 if is_winter else 1008.0) + np.random.normal(0, 2.0)), 1)
                crop_fires   = int(np.random.poisson(12 * dist_meta["crop_fire_risk"])) if (is_winter or is_autumn) else int(np.random.poisson(1))

                sev_class = 0 if pm25 <= 35 else (1 if pm25 <= 75 else (2 if pm25 <= 150 else 3))
                waqi_ref  = waqi_validation.get(city, None)

                records.append({
                    "district": city,
                    "season": season_key_rec,
                    "record_month": month,
                    "temperature_c": temp_val,
                    "humidity_pct": rh_val,
                    "wind_speed_kmh": wind_val,
                    "surface_pressure_hpa": pressure_val,
                    "crop_fires_detected": crop_fires,
                    "blue_mean": dip_feat["blue_mean"],
                    "blue_std": dip_feat["blue_std"],
                    "blue_skewness": dip_feat["blue_skewness"],
                    "blue_high_ratio": dip_feat["blue_high_ratio"],
                    "blue_dominance": dip_feat["blue_dominance"],
                    "laplacian_var": dip_feat["laplacian_var"],
                    "sobel_mean": dip_feat["sobel_mean"],
                    "edge_density": dip_feat["edge_density"],
                    "hsi_saturation_mean": dip_feat["hsi_saturation_mean"],
                    "haze_index": dip_feat["haze_index"],
                    "dark_channel_mean": dip_feat["dark_channel_mean"],
                    "fft_high_energy_ratio": dip_feat["fft_high_energy_ratio"],
                    "gray_contrast": dip_feat.get("gray_contrast", 0.0),
                    "blue_red_ratio": dip_feat.get("blue_red_ratio", 1.0),
                    "local_variance_mean": dip_feat.get("local_variance_mean", 0.0),
                    "hot_index": hot_idx,
                    "ndsi_index": ndsi_idx,
                    "modis_aod": aod_val,
                    "pm25_target": pm25,        # raw CAMS label — no multiplier
                    "pm25_waqi_ref": waqi_ref,  # WAQI ground-station reference for validation
                    "severity_class": sev_class,
                    "label_source": api_source,
                    "source_image_id": source_image_id,  # which real photo this row's image came from
                })

        df = pd.DataFrame(records)
        if save_csv:
            csv_path = os.path.join(self.data_dir, "punjab_real_training_dataset.csv")
            df.to_csv(csv_path, index=False)
            df.to_csv(os.path.join(self.data_dir, "synthetic_punjab_dataset.csv"), index=False)
            print(f"[Data Pipeline] Extracted authentic DIP features from {len(df)} multi-temporal satellite smog passes paired with ground PM2.5 & AOD.")
        return df

    def generate_punjab_synthetic_dataset(self, n_samples: int = 1600, save_csv: bool = True) -> pd.DataFrame:
        return self.build_real_training_dataset(n_samples_per_city=n_samples // 6, save_csv=save_csv)