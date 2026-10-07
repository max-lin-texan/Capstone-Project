from pathlib import Path
from file_read import load_data

DATA_DIR = Path.home() / "Capstone-Project" / "data"

MEDIAN_VARS = [
    "sao2", "heartrate", "respiration",
    "temperature", "gcs", "nibp_systolic", "nibp_diastolic", "nibp_mean",
]

# Physiologically plausible ranges (inclusive); temperature and gcs are already filtered in file_read
VALID_RANGES = {
    "heartrate": (20, 300),
    "sao2": (50, 100),
    "respiration": (1, 80),
    "nibp_systolic": (40, 300),
    "nibp_diastolic": (10, 200),
    "nibp_mean": (20, 250),
}
NIBP = ["nibp_systolic", "nibp_mean", "nibp_diastolic"]

def clean_data(df, p=True):
    """Set implausible values to NaN in place; rows are kept."""
    report = {}
    for v, (lo, hi) in VALID_RANGES.items():
        n_before = df[v].notna().sum()
        bad = df[v].notna() & ~df[v].between(lo, hi)
        df.loc[bad, v] = float("nan")
        report[v] = [n_before, bad.sum(), 0]

    s, m, d = (df[c] for c in NIBP)
    bad_bp = s.notna() & m.notna() & d.notna() & ~((s > m) & (m > d))
    df.loc[bad_bp, NIBP] = float("nan")
    for c in NIBP:
        report[c][2] = bad_bp.sum()

    if p:
        print("Cleaning (values set to NaN):")
        print(f"  {'variable':<15} {'records':>11} {'out of range':>16} {'BP order':>14}")
        for v, (n, r, b) in report.items():
            print(f"  {v:<15} {n:>11,} {r:>9,} ({r / n:.2%}) {b:>7,} ({b / n:.2%})")

    return df

def median_in_window(df, start_h=0, end_h=6, p=True):
    """Per-patient median of each variable within [start_h, end_h) hours after ICU admission."""
    all_ids = df["patientunitstayid"].unique()
    in_window = df["observationoffset"].between(start_h * 60, end_h * 60, inclusive="left")
    g = df.loc[in_window, ["patientunitstayid"] + MEDIAN_VARS].groupby("patientunitstayid")

    medians = g.median()
    counts = g.count().add_prefix("n_")

    df_median = medians.join(counts).reindex(all_ids)
    n_cols = counts.columns
    df_median[n_cols] = df_median[n_cols].fillna(0).astype("int32")
    df_median = df_median.rename_axis("patientunitstayid").reset_index()

    if p:
        n = len(df_median)
        print(f"Window {start_h}-{end_h}h: {n:,} patients")
        for v in MEDIAN_VARS:
            k = df_median[v].notna().sum()
            print(f"  {v:<15} {k:>7,} patients ({k / n:.1%})")
        k = df_median[MEDIAN_VARS].notna().all(axis=1).sum()
        print(f"  all {len(MEDIAN_VARS)} variables: {k:,} patients ({k / n:.1%})")
        k = (df_median[n_cols].sum(axis=1) == 0).sum()
        print(f"  no data in window: {k:,} patients ({k / n:.1%})")

    return df_median

def data_preprocessing():
    df_sepsis_vitals = load_data(p=True)
    df_sepsis_vitals = clean_data(df_sepsis_vitals)
    df_median = median_in_window(df_sepsis_vitals, 0, 6)
    DATA_DIR.mkdir(exist_ok=True)
    df_median.to_csv(DATA_DIR / "df_median.csv", index=False)
    df_median_complete = df_median.dropna(subset=MEDIAN_VARS)
    df_median_complete.to_csv(DATA_DIR / "df_median_complete.csv", index=False)

    return df_median

df_median = data_preprocessing()
print(df_median.head())
