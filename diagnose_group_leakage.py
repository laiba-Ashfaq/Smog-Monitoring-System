"""
Diagnose why Train/Test gap persists after the leak-free split + CV overfitting
penalty (which barely moved the numbers: Train R2 stayed ~0.82 across changes).

Hypothesis being tested: source_image_id groups one calendar day per city, and
weather/AOD readings for that day come from a reanalysis model (CAMS) that varies
smoothly hour-to-hour. If a feature is nearly CONSTANT within a day but varies a
lot BETWEEN days, a tree model can use it as a near-unique "which day is this"
key -- fitting each training day's average target almost exactly (inflating
Train R2 / accuracy) without learning anything that transfers to an unseen day,
even though the day itself never appears in both train and test (the split is
still leak-free; this is a different, subtler problem: effective sample size is
closer to the number of DAYS than the number of ROWS).

This computes each feature's intraclass correlation (ICC): the fraction of a
feature's total variance that is BETWEEN groups rather than within groups.
ICC close to 1.0 = the feature is basically a per-day constant (highest risk).
ICC close to 0.0 = the feature varies as much within a day as across days
(no memorization risk from this feature).

Run this after `py train.py` has produced data/punjab_real_training_dataset.csv.
"""
import pandas as pd
import numpy as np
import os

CSV_PATH = os.path.join("data", "punjab_real_training_dataset.csv")

FEATURE_COLUMNS = [
    "temperature_c", "humidity_pct", "wind_speed_kmh", "surface_pressure_hpa",
    "modis_aod", "crop_fires_detected",
    "blue_mean", "blue_std", "blue_skewness", "blue_high_ratio", "blue_dominance",
    "laplacian_var", "sobel_mean", "edge_density", "hsi_saturation_mean",
    "haze_index", "dark_channel_mean", "fft_high_energy_ratio",
    "hot_index", "ndsi_index",
]


def intraclass_correlation(df: pd.DataFrame, feature: str, group_col: str = "source_image_id") -> float:
    """
    ICC = between-group variance / total variance, using a one-way random-effects
    approximation (standard ANOVA-style ICC). Values near 1.0 mean 'this feature
    is basically a lookup key for which group/day a row belongs to'.
    """
    grouped = df.groupby(group_col)[feature]
    group_means = grouped.transform("mean")
    grand_mean = df[feature].mean()
    total_var = ((df[feature] - grand_mean) ** 2).sum()
    if total_var == 0:
        return float("nan")
    between_var = ((group_means - grand_mean) ** 2).sum()
    return float(between_var / total_var)


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Could not find {CSV_PATH}. Run `py train.py` first so the real dataset is written.")
        return

    df = pd.read_csv(CSV_PATH)
    if "source_image_id" not in df.columns:
        print("No source_image_id column found -- can't check group-level memorization risk.")
        return

    n_groups = df["source_image_id"].nunique()
    rows_per_group = df.groupby("source_image_id").size()
    print(f"Total rows: {len(df)}   Distinct groups (days): {n_groups}   "
          f"Rows/group: min={rows_per_group.min()} mean={rows_per_group.mean():.1f} max={rows_per_group.max()}")
    print()
    print(f"{'feature':<24} {'ICC (between/total var)':>26}   risk")
    print("-" * 65)

    results = []
    for feat in FEATURE_COLUMNS:
        if feat not in df.columns:
            continue
        icc = intraclass_correlation(df, feat)
        results.append((feat, icc))

    results.sort(key=lambda x: (-x[1] if not np.isnan(x[1]) else -1))
    for feat, icc in results:
        if np.isnan(icc):
            risk = "n/a (zero variance)"
        elif icc > 0.90:
            risk = "HIGH  <-- near-constant per day, likely acting as a day-ID"
        elif icc > 0.70:
            risk = "moderate"
        else:
            risk = "low"
        print(f"{feat:<24} {icc:>26.4f}   {risk}")

    print()
    high_risk = [f for f, icc in results if not np.isnan(icc) and icc > 0.90]
    if high_risk:
        print(f"HIGH-RISK features (ICC > 0.90): {high_risk}")
        print("These vary almost entirely BETWEEN days, not within a day. A tree model")
        print("can use them to identify which training day a row came from and reproduce")
        print("that day's average target almost exactly -- inflating Train R2/accuracy")
        print("without learning anything that transfers to an unseen day. This is a")
        print("genuine effective-sample-size problem (~N groups, not N rows), not a fixable")
        print("hyperparameter/CV issue, and it would explain why the gap barely moved when")
        print("the search was told to penalize overfitting.")
    else:
        print("No feature exceeds ICC 0.90 -- day-level memorization via a single feature")
        print("is not the dominant explanation. The gap is more likely genuine capacity/")
        print("data-volume limited (~N groups) rather than one runaway quasi-ID feature.")


if __name__ == "__main__":
    main()