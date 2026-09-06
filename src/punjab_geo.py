"""
Punjab Geospatial & Air Quality Metadata Module
================================================
Defines Punjab district coordinates, vulnerability profiles, ground EPA Punjab stations,
and regulatory smog severity standards for Lahore, Multan, Faisalabad, Rawalpindi,
and other critical Punjab districts.
"""

from typing import Dict, List, Any

# Complete Punjab Districts Metadata (including all 4 primary training cities: Lahore, Multan, Faisalabad, Rawalpindi)
PUNJAB_DISTRICTS: Dict[str, Dict[str, Any]] = {
    "Lahore": {
        "lat": 31.5204,
        "lon": 74.3587,
        "population": "13.9M",
        "category": "High Vulnerability (Capital)",
        "primary_sources": ["Vehicular (43%)", "Industrial (25%)", "Crop Burning Transboundary (22%)", "Waste Burning (10%)"],
        "baseline_summer_pm25": 45.0,
        "baseline_winter_pm25": 280.0,
        "crop_fire_risk": 0.85,
        "mgrs_tile": "42RWU"
    },
    "Multan": {
        "lat": 30.1575,
        "lon": 71.5249,
        "population": "2.2M",
        "category": "South Punjab Agricultural Hub",
        "primary_sources": ["Cotton Processing", "Thermal Inversion", "Vehicular", "Crop Burning"],
        "baseline_summer_pm25": 32.0,
        "baseline_winter_pm25": 190.0,
        "crop_fire_risk": 0.65,
        "mgrs_tile": "42RVT"
    },
    "Faisalabad": {
        "lat": 31.4504,
        "lon": 73.1350,
        "population": "3.8M",
        "category": "Industrial Textile Center",
        "primary_sources": ["Textile Mills", "Power Generation", "Vehicular", "Brick Kilns"],
        "baseline_summer_pm25": 40.0,
        "baseline_winter_pm25": 230.0,
        "crop_fire_risk": 0.70,
        "mgrs_tile": "42RUN"
    },
    "Rawalpindi": {
        "lat": 33.5651,
        "lon": 73.0169,
        "population": "2.3M",
        "category": "Potohar Plateau Zone",
        "primary_sources": ["Vehicular Traffic", "Construction Dust", "Topographical Trapping"],
        "baseline_summer_pm25": 25.0,
        "baseline_winter_pm25": 115.0,
        "crop_fire_risk": 0.25,
        "mgrs_tile": "42RWV"
    },
    "Sheikhupura": {
        "lat": 31.7131,
        "lon": 73.9783,
        "population": "1.7M",
        "category": "Paddy Crop Burning Epicenter",
        "primary_sources": ["Intense Paddy Stubble Burning", "Rice Mills", "Brick Kilns"],
        "baseline_summer_pm25": 38.0,
        "baseline_winter_pm25": 320.0,
        "crop_fire_risk": 0.98,
        "mgrs_tile": "42RWU"
    },
    "Gujranwala": {
        "lat": 32.1877,
        "lon": 74.1945,
        "population": "2.4M",
        "category": "Foundry & Metal Corridor",
        "primary_sources": ["Foundries & Metal Smelting", "Stubble Burning", "Freight Transport"],
        "baseline_summer_pm25": 42.0,
        "baseline_winter_pm25": 260.0,
        "crop_fire_risk": 0.90,
        "mgrs_tile": "42RWU"
    },
    "Kasur": {
        "lat": 31.1179,
        "lon": 74.4500,
        "population": "1.4M",
        "category": "Border Stubble Buffer Zone",
        "primary_sources": ["Transboundary Smoke", "Tanneries", "Agricultural Residue"],
        "baseline_summer_pm25": 36.0,
        "baseline_winter_pm25": 275.0,
        "crop_fire_risk": 0.88,
        "mgrs_tile": "42RWU"
    },
    "Sialkot": {
        "lat": 32.4945,
        "lon": 74.5229,
        "population": "1.1M",
        "category": "Industrial Export Belt",
        "primary_sources": ["Surgical & Leather Goods Units", "Stubble Drift", "Vehicular"],
        "baseline_summer_pm25": 30.0,
        "baseline_winter_pm25": 185.0,
        "crop_fire_risk": 0.65,
        "mgrs_tile": "42RWU"
    },
    "Sahiwal": {
        "lat": 30.6682,
        "lon": 73.1114,
        "population": "1.3M",
        "category": "Agricultural Corridor",
        "primary_sources": ["Coal Power Plant", "Crop Residue Burning"],
        "baseline_summer_pm25": 28.0,
        "baseline_winter_pm25": 195.0,
        "crop_fire_risk": 0.75,
        "mgrs_tile": "42RUN"
    },
    "Bahawalpur": {
        "lat": 29.3544,
        "lon": 71.6911,
        "population": "0.9M",
        "category": "Arid Southern Belt",
        "primary_sources": ["Desert Dust", "Agricultural Waste"],
        "baseline_summer_pm25": 35.0,
        "baseline_winter_pm25": 140.0,
        "crop_fire_risk": 0.40,
        "mgrs_tile": "42RVT"
    }
}

# Ground Truth EPA Punjab Air Quality Monitoring Stations across Lahore, Multan, Faisalabad, Rawalpindi
PEQS_STATIONS: List[Dict[str, Any]] = [
    {"id": "PEQS-LHR-01", "name": "Town Hall / Lower Mall, Lahore", "district": "Lahore", "lat": 31.5657, "lon": 74.3086, "sensor_type": "Beta Attenuation Monitor (BAM-1020)"},
    {"id": "PEQS-LHR-02", "name": "Gulberg III EPA Head Office, Lahore", "district": "Lahore", "lat": 31.5126, "lon": 74.3483, "sensor_type": "Teledyne T640 PM Analyzer"},
    {"id": "PEQS-LHR-03", "name": "Punjab University New Campus, Lahore", "district": "Lahore", "lat": 31.4988, "lon": 74.3025, "sensor_type": "Optical Particle Counter"},
    {"id": "PEQS-LHR-04", "name": "US Consulate Regional Sensor, Lahore", "district": "Lahore", "lat": 31.5497, "lon": 74.3436, "sensor_type": "MetOne BAM Reference Sensor"},
    {"id": "PEQS-MUL-01", "name": "Multan Chungi No. 9 EPA Station", "district": "Multan", "lat": 30.2012, "lon": 71.4720, "sensor_type": "EPA Continuous PM2.5 Monitor"},
    {"id": "PEQS-MUL-02", "name": "Nishtar Hospital Pulmonology Sensor", "district": "Multan", "lat": 30.1870, "lon": 71.4420, "sensor_type": "Aeroqual AQY Micro-station"},
    {"id": "PEQS-FSD-01", "name": "Faisalabad EPA Complex, D-Ground", "district": "Faisalabad", "lat": 31.4180, "lon": 73.0790, "sensor_type": "Horiba APDA-372 Continuous Analyzer"},
    {"id": "PEQS-FSD-02", "name": "Agriculture University Faisalabad (UAF)", "district": "Faisalabad", "lat": 31.4310, "lon": 73.0690, "sensor_type": "MetOne E-BAM"},
    {"id": "PEQS-RWP-01", "name": "Rawalpindi Committee Chowk Station", "district": "Rawalpindi", "lat": 33.6060, "lon": 73.0680, "sensor_type": "EPA Mobile Air Lab Monitor"},
    {"id": "PEQS-SKP-01", "name": "Sheikhupura Industrial Area Station", "district": "Sheikhupura", "lat": 31.7190, "lon": 73.9850, "sensor_type": "PEQS Particulate Station"},
    {"id": "PEQS-GRW-01", "name": "Gujranwala Chamber of Commerce", "district": "Gujranwala", "lat": 32.1602, "lon": 74.1850, "sensor_type": "PEQS Ambient Sensor"}
]

# Smog Severity Definitions (4 Classes)
SEVERITY_LEVELS: Dict[int, Dict[str, Any]] = {
    0: {
        "name": "Clean / Good",
        "pm25_min": 0.0,
        "pm25_max": 35.0,
        "aqi_min": 0,
        "aqi_max": 50,
        "color": "#10b981",
        "hex": "#10b981",
        "description": "Air quality is satisfactory, poses little or no health risk.",
        "action": "Ideal for all outdoor activities. No restrictions."
    },
    1: {
        "name": "Moderate Haze",
        "pm25_min": 35.1,
        "pm25_max": 75.0,
        "aqi_min": 51,
        "aqi_max": 150,
        "color": "#f59e0b",
        "hex": "#f59e0b",
        "description": "Acceptable air quality; sensitive groups may experience minor irritation.",
        "action": "Sensitive individuals (asthma, children) should limit prolonged outdoor exertion."
    },
    2: {
        "name": "Unhealthy / Dense Smog",
        "pm25_min": 75.1,
        "pm25_max": 150.0,
        "aqi_min": 151,
        "aqi_max": 300,
        "color": "#ef4444",
        "hex": "#ef4444",
        "description": "Everyone may begin to experience adverse health effects. Serious for sensitive groups.",
        "action": "Schools limit outdoor sports. N95 masks strongly recommended outdoors."
    },
    3: {
        "name": "Hazardous / Emergency Smog",
        "pm25_min": 150.1,
        "pm25_max": 800.0,
        "aqi_min": 301,
        "aqi_max": 800,
        "color": "#7f1d1d",
        "hex": "#7f1d1d",
        "description": "Health emergency warning. Entire population is more likely to be severely affected.",
        "action": "School closures triggered. Work-from-home mandate, brick kilns & stubble burning locked down by EPA."
    }
}
