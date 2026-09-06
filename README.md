# Smog Sentinel Punjab - AI-Powered Smog Monitoring System

## Project Overview

Smog Sentinel Punjab is a real-time, AI-powered smog monitoring and early-warning system built for the Alibaba Cloud AI Hackathon. It targets one of South Asia's most pressing environmental health crises: the seasonal smog that blankets Punjab, Pakistan, particularly during the crop-burning months, severely affecting air quality in Lahore, Faisalabad, Sheikhupura, Gujranwala, Multan, and Rawalpindi.

The system fuses satellite imagery, weather data, ground-station readings, and active-fire data into a single pipeline that classifies smog severity, estimates PM2.5 concentration, visualizes results on an interactive map, and automatically alerts the stakeholders best positioned to act on the information.

This project demonstrates the integration of classical digital image processing, gradient-boosted machine learning, geospatial dashboards, and cloud deployment to turn freely available satellite data into actionable public-health intelligence at a fraction of the cost of physical sensor networks.

---

## Objectives

* Provide real-time, district-level smog severity and PM2.5 estimates across Punjab using satellite imagery instead of relying solely on sparse ground sensors.
* Detect and localize crop-residue burning, a major seasonal driver of smog, using active-fire satellite data.
* Deliver an interactive, map-based dashboard that authorities and the public can use to understand current and historical air quality.
* Automate multi-stakeholder alerting so schools, hospitals, enforcement agencies, and the public are notified before conditions become hazardous.
* Demonstrate a leak-free, rigorously validated machine learning pipeline ready for deployment on Alibaba Cloud.

---

## Features

* Live, color-coded smog severity heatmap across six Punjab districts.
* Dual CatBoost models: a 4-class severity classifier (Good / Moderate / Unhealthy / Hazardous) and a continuous PM2.5 regressor.
* Classical Digital Image Processing (DIP) engine extracting blue-channel histogram, Laplacian edge-sharpness, HSI saturation, and 2D FFT features from imagery.
* Real-time image analyzer: upload a satellite, drone, or CCTV image and see it decomposed into the four DIP diagnostic maps.
* NASA FIRMS crop-fire hotspot overlay with fire radiative power (FRP) metrics.
* Historical trend views over 30, 90, and 365-day windows, with correlation analysis between weather, fires, and pollution levels.
* Automated, threshold-based alert engine issuing bilingual (Urdu/English) notifications to schools, hospitals/Rescue 1122, EPA enforcement squads, and the public.
* Strict 70/30 train-test split stratified by district and severity, with an explicit zero-data-leakage check and an automated overfitting audit.
* One-click deployment tooling for Alibaba Cloud (ECS, OSS, and PAI for retraining).

---

## Technologies Used

| Technology | Purpose |
| --- | --- |
| Python | Core language for the entire pipeline |
| OpenCV / NumPy | Classical DIP feature extraction (histogram, edge, HSI, FFT) |
| CatBoost | Gradient-boosted severity classifier and PM2.5 regressor |
| Streamlit | Interactive dashboard framework |
| Folium | Geospatial map rendering |
| Plotly | Trend charts and correlation visualizations |
| SQLite | Local persistence for readings and alerts |
| Docker | Containerized deployment |
| Alibaba Cloud (ECS, OSS, PAI) | Hosting, storage, and automated retraining |

---

## System Architecture

The system consists of:

* A data ingestion layer pulling ESA Sentinel-2 imagery, NASA data, Open-Meteo/ECMWF weather, Copernicus CAMS air quality data, EPA Punjab ground-station readings, and NASA FIRMS fire hotspots.
* A classical DIP feature-extraction module that converts raw imagery into physically interpretable haze indicators.
* A dual CatBoost model layer producing severity classification and PM2.5 regression outputs.
* A morphological post-processing step that smooths the boundaries of detected smog plumes.
* A Streamlit + Folium + Plotly dashboard for visualization, and an alert engine for automated notifications.
* A deployment layer (Docker + Alibaba Cloud scripts) for taking the system from local development to public hosting.

### Workflow

1. The system ingests satellite imagery, weather data, ground-station readings, and fire hotspot data for each monitored district.
2. The DIP engine extracts histogram, edge, color-saturation, and frequency-domain features from the imagery.
3. The CatBoost models consume these features (plus weather and fire context) to classify severity and estimate PM2.5.
4. Morphological post-processing cleans up the resulting smog-severity map.
5. Results are rendered on the live dashboard, alongside historical trends and fire hotspot overlays.
6. The alert engine checks results against thresholds and triggers bilingual notifications to the relevant stakeholders.

---

## Validated Performance

All metrics below come from a strict 70/30 train-test split stratified by district and severity class, with zero district overlap between splits, and an automated overfitting audit that returned a **PASSED** verdict:

| Metric | Result |
| --- | --- |
| PM2.5 regression R² (test) | 0.8576 (train: 0.9254, gap: 0.068) |
| Mean Absolute Error (PM2.5) | 10.94 µg/m³ |
| Root Mean Square Error (PM2.5) | 15.91 µg/m³ |
| Smog area IoU (Moderate or worse) | 0.753 |
| Dense smog IoU (Unhealthy or worse) | 0.560 |

---

## Screenshots

> ### Dashboard - Smog Severity Map
> <img width="1323" height="1600" alt="Image" src="https://github.com/user-attachments/assets/435fc018-9e1d-4077-8fd9-c26b2694fc50" />

> ### Real-Time Image Analyzer
> <img width="800" height="1529" alt="Image" src="https://github.com/user-attachments/assets/316ce822-f15a-47ae-9564-a30f74cb82af" />

> ### Historical Trends View
> <img width="1329" height="1600" alt="Image" src="https://github.com/user-attachments/assets/834d6d7f-3a63-45eb-be6e-470db431a838" />

---

## Repository Structure

```text
smog_sentinel_punjab/
│
├── README.md
├── requirements.txt
├── app.py                     # Streamlit dashboard entry point
├── api.py                     # API layer
├── train.py                   # End-to-end training pipeline
├── database.py                # SQLite persistence layer
├── run_api.bat                # Windows launcher for the API
├── run_dashboard.bat          # Windows launcher for the dashboard
│
├── src/
│   ├── dip_extractor.py       # Classical DIP feature extraction
│   ├── ml_pipeline.py         # CatBoost models + post-processing
│   ├── data_ingestion.py      # Weather, fire, and ground-station ingestion
│   ├── alert_engine.py        # Threshold-based alert rules
│   ├── sample_generator.py    # Synthetic/sample patch generation
│   └── punjab_geo.py          # District coordinates and station metadata
│
├── data/
│   ├── sample_patches/        # Labeled sample images for the analyzer
│   ├── punjab_real_training_dataset.csv
│   └── synthetic_punjab_dataset.csv
│
├── models/
│   ├── severity_classifier.cbm
│   ├── severity_classifier_catboost.joblib
│   ├── pm25_regressor.joblib
│   └── metrics.json
│
├── deployment/
│   ├── Dockerfile
│   ├── alibaba_deploy.sh      # One-click Alibaba Cloud ECS deployment
│   └── oss_sync.py            # Sync imagery/models to Alibaba OSS
│
├── diagnose_group_leakage.py  # Data-leakage diagnostic script
├── test_image_detection.py    # Image-detection accuracy tests
└── verify_fixes.py            # Verification script for bug fixes
```

---

## How to Run

### Training

1. Navigate to the project root.
2. Run `pip install -r requirements.txt` to install dependencies.
3. Run `python train.py` to generate sample patches, build the calibrated dataset, and train both models with leak-free validation.

### Dashboard

1. Run `streamlit run app.py` (or `run_dashboard.bat` on Windows).
2. Open the dashboard in your browser to view the live severity map, fire hotspots, and image analyzer.

### API

1. Run `python api.py` (or `run_api.bat` on Windows) to start the API server.

### Deployment

1. Build the Docker image using the provided `Dockerfile`.
2. Run `deployment/alibaba_deploy.sh` for one-click deployment to Alibaba Cloud ECS.
3. Use `deployment/oss_sync.py` to sync imagery and trained models to Alibaba Cloud OSS.

---

## Applications

* Environmental Protection Agencies (smog enforcement, crop-fire evidence)
* Schools and the Education Department (closure decisions)
* Hospitals and Emergency Services (patient preparedness)
* Urban Planners (long-term air-quality-informed decisions)
* Aviation Authorities (visibility-informed flight operations)

---

## Future Enhancements

* Native iOS/Android app built on top of the existing Streamlit dashboard.
* Live deployment of the already-trained, 6-district models to Alibaba Cloud ECS/OSS for public 24/7 access.
* National-scale air quality platform covering all of Pakistan.
* Legally admissible crop-fire evidence generation for EPA enforcement.
* Official integration with the Pakistan Meteorological Department's reporting.
* Smog forecasting by combining current smog maps with weather forecasts.

---

## License

This project is developed for educational and hackathon purposes. No formal open-source license has been applied yet.
