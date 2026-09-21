import pandas as pd
from pathlib import Path

path = Path.home() / "eICU"

df1 = pd.read_csv(path / "admissionDx.csv")
df2 = pd.read_csv(path / "diagnosis.csv")
# Read IDs only to check presence in vitalPeriodic
df3_ids = pd.read_csv(path / "vitalPeriodic.csv", usecols=["patientunitstayid"])

ids3 = set(df3_ids["patientunitstayid"].unique())

# Find sepsis in either diagnosis table (union)
sepsis_pat = r"sepsis|septicemia|septic\b"
icd_pat = r"(?:^|[, ])(?:038(?:\.\d+)?|995\.9[12]|A40(?:\.\d+)?|A41(?:\.\d+)?)(?:$|[, ])"

sepsis_from_df1 = set(
    df1.loc[
        df1["admitdxpath"].fillna("").str.contains(sepsis_pat, case=False, regex=True)
        | df1["admitdxname"].fillna("").str.contains(sepsis_pat, case=False, regex=True)
        | df1["admitdxtext"].fillna("").str.contains(sepsis_pat, case=False, regex=True),
        "patientunitstayid",
    ].unique()
)

sepsis_from_df2 = set(
    df2.loc[
        df2["diagnosisstring"].fillna("").str.contains(sepsis_pat, case=False, regex=True)
        | df2["icd9code"].fillna("").astype(str).str.contains(icd_pat, case=False, regex=True),
        "patientunitstayid",
    ].unique()
)

# Patients with sepsis in either diagnosis table
sepsis_patients = sepsis_from_df1 | sepsis_from_df2

# Keep only those who also appear in vitalPeriodic
sepsis_with_vitals = sepsis_patients & ids3

print(f"sepsis in admissionDx: {len(sepsis_from_df1):,}")
print(f"sepsis in diagnosis:   {len(sepsis_from_df2):,}")
print(f"sepsis in either table (union): {len(sepsis_patients):,}")
print(f"of those, also in vitalPeriodic: {len(sepsis_with_vitals):,}")
