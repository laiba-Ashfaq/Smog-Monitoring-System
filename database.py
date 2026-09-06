"""
SQLite Time-Series Storage & Database Layer
===========================================
Persists continuous sensor readings, satellite inferences, weather telemetry,
threshold alert logs, and NASA FIRMS active crop fire hotspots using SQLAlchemy / SQLite.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, desc
from sqlalchemy.orm import declarative_base, sessionmaker, Session

Base = declarative_base()

class DistrictReading(Base):
    """Hourly & daily district air quality and meteorological telemetry."""
    __tablename__ = "district_readings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    district = Column(String(50), index=True, nullable=False)
    pm25 = Column(Float, nullable=False)
    severity_class = Column(Integer, nullable=False)
    severity_name = Column(String(50), nullable=False)
    temperature_c = Column(Float, nullable=False)
    humidity_pct = Column(Float, nullable=False)
    wind_speed_kmh = Column(Float, nullable=False)
    pressure_hpa = Column(Float, nullable=False)
    crop_fires = Column(Integer, default=0)
    hot_index = Column(Float, nullable=True)
    ndsi_index = Column(Float, nullable=True)
    laplacian_var = Column(Float, nullable=True)
    blue_mean = Column(Float, nullable=True)

class SatelliteInferenceLog(Base):
    """Audit log of image patches processed by DIP + CatBoost/XGBoost models."""
    __tablename__ = "satellite_inferences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    patch_id = Column(String(100), nullable=False)
    district = Column(String(50), nullable=False)
    predicted_pm25 = Column(Float, nullable=False)
    severity_class = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    hot_index = Column(Float, nullable=True)
    laplacian_var = Column(Float, nullable=True)
    blue_mean = Column(Float, nullable=True)
    hsi_sat = Column(Float, nullable=True)
    fft_high_ratio = Column(Float, nullable=True)

class AlertLog(Base):
    """Audit log for automated threshold triggers and SMS delivery receipts."""
    __tablename__ = "alert_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    alert_id = Column(String(100), nullable=False)
    district = Column(String(50), nullable=False)
    tier = Column(String(100), nullable=False)
    pm25 = Column(Float, nullable=False)
    stakeholders = Column(String(255), nullable=False)
    sms_english = Column(Text, nullable=False)
    sms_urdu = Column(Text, nullable=False)
    recipient_number = Column(String(50), default="+92 300 0000000")
    status = Column(String(50), default="DELIVERED")
    latency_ms = Column(Integer, default=140)

class CropFireRecord(Base):
    """NASA FIRMS active thermal anomalies and crop residue burning detections."""
    __tablename__ = "crop_fire_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fire_id = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cluster = Column(String(100), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    frp_mw = Column(Float, nullable=False)
    brightness_k = Column(Float, nullable=False)
    confidence = Column(String(20), default="High")
    satellite = Column(String(50), default="SNPP-VIIRS")


class TimeSeriesDatabase:
    """Manages SQLite time-series storage and analytical queries."""

    def __init__(self, db_path: str = "data/smog_sentinel.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.seed_historical_if_empty()

    def get_session(self) -> Session:
        return self.SessionLocal()

    def seed_historical_if_empty(self) -> None:
        """Seeds 365 days of historical time-series data across Punjab districts if DB is empty."""
        with self.get_session() as session:
            count = session.query(DistrictReading).count()
            if count > 0:
                return

            print("[Database] Seeding 365 days of time-series records into SQLite...")
            from src.punjab_geo import PUNJAB_DISTRICTS, SEVERITY_LEVELS
            
            records = []
            base_date = datetime.now() - timedelta(days=365)
            np.random.seed(42)

            for day_offset in range(365):
                current_time = base_date + timedelta(days=day_offset, hours=12)
                month = current_time.month
                is_winter = month in [11, 12, 1]
                is_autumn = month in [9, 10]
                is_monsoon = month in [7, 8]

                for district_name, dist_meta in PUNJAB_DISTRICTS.items():
                    # Seasonal weather parameters
                    if is_winter:
                        temp = 14.0 + np.random.normal(0, 3)
                        humid = 82.0 + np.random.normal(0, 8)
                        wind = 3.8 + np.random.normal(0, 1.5)
                        crop_fires = int(np.random.poisson(15 * dist_meta["crop_fire_risk"]))
                        base_pm25 = dist_meta["baseline_winter_pm25"] * (0.8 + 0.4 * np.random.uniform(0.7, 1.3))
                    elif is_autumn:
                        temp = 24.0 + np.random.normal(0, 3)
                        humid = 60.0 + np.random.normal(0, 7)
                        wind = 5.5 + np.random.normal(0, 2)
                        crop_fires = int(np.random.poisson(30 * dist_meta["crop_fire_risk"]))
                        base_pm25 = dist_meta["baseline_winter_pm25"] * 0.75 + crop_fires * 2.2
                    elif is_monsoon:
                        temp = 32.0 + np.random.normal(0, 2)
                        humid = 80.0 + np.random.normal(0, 8)
                        wind = 14.0 + np.random.normal(0, 4)
                        crop_fires = 0
                        base_pm25 = dist_meta["baseline_summer_pm25"] * 0.6
                    else:  # Summer
                        temp = 38.0 + np.random.normal(0, 4)
                        humid = 35.0 + np.random.normal(0, 6)
                        wind = 11.0 + np.random.normal(0, 3)
                        crop_fires = int(np.random.poisson(2))
                        base_pm25 = dist_meta["baseline_summer_pm25"]

                    pm25 = max(8.0, round(float(base_pm25 + np.random.normal(0, 10)), 1))

                    if pm25 <= 35.0:
                        sev_class = 0
                    elif pm25 <= 75.0:
                        sev_class = 1
                    elif pm25 <= 150.0:
                        sev_class = 2
                    else:
                        sev_class = 3

                    sev_name = SEVERITY_LEVELS[sev_class]["name"]

                    # DIP indices
                    hot_idx = round(float(0.05 + (pm25 / 500.0) * 0.35 + np.random.normal(0, 0.02)), 3)
                    ndsi_idx = round(float(0.10 + (pm25 / 500.0) * 0.40 + np.random.normal(0, 0.03)), 3)
                    lap_var = round(float(max(15.0, 600.0 - (pm25 / 500.0) * 480.0 + np.random.normal(0, 20))), 1)
                    blue_mean = round(float(min(245.0, 95.0 + (pm25 / 500.0) * 120.0 + np.random.normal(0, 6))), 1)

                    records.append(DistrictReading(
                        timestamp=current_time,
                        district=district_name,
                        pm25=pm25,
                        severity_class=sev_class,
                        severity_name=sev_name,
                        temperature_c=round(temp, 1),
                        humidity_pct=round(humid, 1),
                        wind_speed_kmh=round(max(1.0, wind), 1),
                        pressure_hpa=round(1014.0 + np.random.normal(0, 3), 1),
                        crop_fires=crop_fires,
                        hot_index=hot_idx,
                        ndsi_index=ndsi_idx,
                        laplacian_var=lap_var,
                        blue_mean=blue_mean
                    ))

            session.bulk_save_objects(records)
            session.commit()
            print(f"[Database] Successfully seeded {len(records)} time-series rows to SQLite!")

    def insert_reading(self, reading_data: Dict[str, Any]) -> int:
        """Inserts a single new district sensor/weather reading."""
        with self.get_session() as session:
            obj = DistrictReading(
                timestamp=reading_data.get("timestamp", datetime.utcnow()),
                district=reading_data["district"],
                pm25=reading_data["pm25"],
                severity_class=reading_data["severity_class"],
                severity_name=reading_data["severity_name"],
                temperature_c=reading_data.get("temperature_c", 20.0),
                humidity_pct=reading_data.get("humidity_pct", 60.0),
                wind_speed_kmh=reading_data.get("wind_speed_kmh", 8.0),
                pressure_hpa=reading_data.get("pressure_hpa", 1013.0),
                crop_fires=reading_data.get("crop_fires", 0),
                hot_index=reading_data.get("hot_index", 0.1),
                ndsi_index=reading_data.get("ndsi_index", 0.1),
                laplacian_var=reading_data.get("laplacian_var", 300.0),
                blue_mean=reading_data.get("blue_mean", 120.0),
            )
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return obj.id

    def insert_inference(self, inference_data: Dict[str, Any]) -> int:
        """Logs a satellite image patch inference."""
        with self.get_session() as session:
            obj = SatelliteInferenceLog(
                timestamp=inference_data.get("timestamp", datetime.utcnow()),
                patch_id=inference_data["patch_id"],
                district=inference_data["district"],
                predicted_pm25=inference_data["predicted_pm25"],
                severity_class=inference_data["severity_class"],
                confidence=inference_data["confidence"],
                hot_index=inference_data.get("hot_index"),
                laplacian_var=inference_data.get("laplacian_var"),
                blue_mean=inference_data.get("blue_mean"),
                hsi_sat=inference_data.get("hsi_sat"),
                fft_high_ratio=inference_data.get("fft_high_ratio"),
            )
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return obj.id

    def insert_alert_log(self, alert_data: Dict[str, Any]) -> int:
        """Logs an alert dispatch event."""
        with self.get_session() as session:
            obj = AlertLog(
                timestamp=alert_data.get("timestamp", datetime.utcnow()),
                alert_id=alert_data["alert_id"],
                district=alert_data["district"],
                tier=alert_data["tier"],
                pm25=alert_data["pm25"],
                stakeholders=str(alert_data["stakeholders"]),
                sms_english=alert_data["sms_english"],
                sms_urdu=alert_data["sms_urdu"],
                recipient_number=alert_data.get("recipient_number", "+92 300 0000000"),
                status=alert_data.get("status", "DELIVERED"),
                latency_ms=alert_data.get("latency_ms", 142)
            )
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return obj.id

    def query_district_history(self, district_name: Optional[str] = None, days: int = 30) -> pd.DataFrame:
        """Queries time-series history for a district or all districts over N days."""
        cutoff = datetime.now() - timedelta(days=days)
        with self.get_session() as session:
            query = session.query(DistrictReading).filter(DistrictReading.timestamp >= cutoff)
            if district_name and district_name != "All Punjab":
                query = query.filter(DistrictReading.district == district_name)
            query = query.order_by(DistrictReading.timestamp.asc())
            
            results = query.all()
            if not results:
                return pd.DataFrame()
                
            data = [
                {
                    "timestamp": r.timestamp,
                    "district": r.district,
                    "pm25": r.pm25,
                    "severity_class": r.severity_class,
                    "severity_name": r.severity_name,
                    "temperature_c": r.temperature_c,
                    "humidity_pct": r.humidity_pct,
                    "wind_speed_kmh": r.wind_speed_kmh,
                    "crop_fires": r.crop_fires,
                    "hot_index": r.hot_index,
                    "laplacian_var": r.laplacian_var
                }
                for r in results
            ]
            return pd.DataFrame(data)

    def query_recent_alerts(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Queries recent alert dispatch logs."""
        with self.get_session() as session:
            rows = session.query(AlertLog).order_by(desc(AlertLog.timestamp)).limit(limit).all()
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "alert_id": r.alert_id,
                    "district": r.district,
                    "tier": r.tier,
                    "pm25": r.pm25,
                    "stakeholders": r.stakeholders,
                    "sms_english": r.sms_english,
                    "sms_urdu": r.sms_urdu,
                    "recipient": r.recipient_number,
                    "status": r.status,
                    "latency_ms": r.latency_ms
                }
                for r in rows
            ]
