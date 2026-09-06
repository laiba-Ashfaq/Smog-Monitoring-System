import sys
sys.path.insert(0, '.')
from src.data_ingestion import DataIngestionPipeline
import inspect

def chk(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")

print("--- build_real_training_dataset checks ---")
src = inspect.getsource(DataIngestionPipeline.build_real_training_dataset)
chk("No i%4 quota forcing", "i % 4" not in src)
chk("No winter multiplier *1.8", "* 1.8" not in src)
chk("No autumn multiplier *1.2", "* 1.2" not in src)
chk("month_to_season used for season", "month_to_season" in src)
chk("raw_pm25 used as PM25 label", "raw_pm25" in src)
chk("pm25_waqi_ref in output records", "pm25_waqi_ref" in src)
chk("label_source in output records", "label_source" in src)
chk("waqi_validation fetched and used", "waqi_validation" in src)

print("--- fetch_peqs_station_readings checks ---")
src2 = inspect.getsource(DataIngestionPipeline.fetch_peqs_station_readings)
chk("No np.random.normal noise injection", "np.random.normal" not in src2)
chk("data_source field present", "data_source" in src2)
chk("raw reading stored directly", "raw_pm25" in src2)

print("--- Module docstring checks ---")
doc = DataIngestionPipeline.__doc__
chk("ERA5 Reanalysis Dataset overclaim removed", "ERA5 Atmospheric Reanalysis Dataset" not in doc)
chk("Honest wording: not ERA5 archive", "not ERA5 archive" in doc)

print("Done.")
