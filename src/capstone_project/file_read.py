import pandas as pd
from functools import reduce
from pathlib import Path

KEYS = ["patientunitstayid", "observationoffset"]
DROP_COLS = ["vitalperiodicid", "temperature", "etco2", "pasystolic", "padiastolic", "pamean", "icp"]
NIBP_COLS = {
    "noninvasivesystolic": "nibp_systolic",
    "noninvasivediastolic": "nibp_diastolic",
    "noninvasivemean": "nibp_mean",
}
URINE_PAT = r"urine|foley|void"
URINE_EXCLUDE_PAT = r"count|occurrence|incontinen|number|unmeasured|mixed"


def _agg(parts, how):
    df = pd.concat(parts, ignore_index=True)
    return df.groupby(KEYS, as_index=False).agg(how)


def _load_nurse_charting(path, ids):
    """Temperature (Celsius) and GCS total from nurseCharting."""
    cols = ["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevallabel",
            "nursingchartcelltypevalname", "nursingchartvalue"]
    temp_parts, gcs_parts = [], []
    info = {"converted": 0, "dropped": 0}
    for chunk in pd.read_csv(path / "nurseCharting.csv", usecols=cols, chunksize=2_000_000,
                             dtype={"nursingchartvalue": str}):
        chunk = chunk[chunk["patientunitstayid"].isin(ids)]
        label = chunk["nursingchartcelltypevallabel"]
        name = chunk["nursingchartcelltypevalname"]

        t = chunk[(label == "Temperature") & name.isin(["Temperature (C)", "Temperature (F)"])]
        v = pd.to_numeric(t["nursingchartvalue"], errors="coerce")
        is_f = v > 50  # no Celsius body temperature exceeds 50, regardless of label
        info["converted"] += int(is_f.sum())
        v = v.where(~is_f, (v - 32) * 5 / 9)
        keep = v.between(25, 45)
        info["dropped"] += int((~keep).sum())
        temp_parts.append(pd.DataFrame({
            "patientunitstayid": t.loc[keep, "patientunitstayid"],
            "observationoffset": t.loc[keep, "nursingchartoffset"],
            "temperature": v[keep],
        }))

        g = chunk[(label == "Glasgow coma score") & (name == "GCS Total")]
        gv = pd.to_numeric(g["nursingchartvalue"], errors="coerce")
        keep = gv.between(3, 15)
        gcs_parts.append(pd.DataFrame({
            "patientunitstayid": g.loc[keep, "patientunitstayid"],
            "observationoffset": g.loc[keep, "nursingchartoffset"],
            "gcs": gv[keep],
        }))

    # C and F rows of the same measurement collapse into one after averaging
    return _agg(temp_parts, "mean"), _agg(gcs_parts, "mean"), info


def _load_nibp(path, ids):
    parts = []
    for chunk in pd.read_csv(path / "vitalAperiodic.csv", usecols=["patientunitstayid", "observationoffset", *NIBP_COLS],
                             chunksize=2_000_000):
        chunk = chunk[chunk["patientunitstayid"].isin(ids)].rename(columns=NIBP_COLS)
        parts.append(chunk.dropna(subset=list(NIBP_COLS.values()), how="all"))
    return _agg(parts, "mean")


def _load_urine(path, ids):
    parts = []
    for chunk in pd.read_csv(path / "intakeOutput.csv",
                             usecols=["patientunitstayid", "intakeoutputoffset", "celllabel", "cellvaluenumeric"],
                             chunksize=2_000_000):
        chunk = chunk[chunk["patientunitstayid"].isin(ids) & (chunk["cellvaluenumeric"] >= 0)]
        label = chunk["celllabel"].fillna("")
        is_urine = (label.str.contains(URINE_PAT, case=False, regex=True)
                    & ~label.str.contains(URINE_EXCLUDE_PAT, case=False, regex=True))
        u = chunk[is_urine]
        parts.append(pd.DataFrame({
            "patientunitstayid": u["patientunitstayid"],
            "observationoffset": u["intakeoutputoffset"],
            "urine_output": u["cellvaluenumeric"],
        }))
    # volumes charted at the same time are added, not averaged
    return _agg(parts, "sum")


def load_patient_info(ids=None):
    """One row per patient: demographics and outcomes."""
    cols = ["patientunitstayid", "age", "gender", "admissionweight",
            "unitdischargestatus", "hospitaldischargestatus", "unitdischargeoffset"]
    df = pd.read_csv(Path.home() / "eICU" / "patient.csv", usecols=cols, dtype={"age": str})
    if ids is not None:
        df = df[df["patientunitstayid"].isin(ids)]
    # eICU records ages above 89 as "> 89"
    df["age"] = pd.to_numeric(df["age"].replace("> 89", "90"), errors="coerce")
    return df.reset_index(drop=True)


def load_data(p = True):
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

    if p:
        print(f"sepsis in admissionDx: {len(sepsis_from_df1):,}")
        print(f"sepsis in diagnosis:   {len(sepsis_from_df2):,}")
        print(f"sepsis in either table (union): {len(sepsis_patients):,}")
        print(f"of those, also in vitalPeriodic: {len(sepsis_with_vitals):,}")

    # Extract vitalPeriodic rows for sepsis patients (chunked to limit memory)
    del df1, df2, df3_ids  # free memory before second pass
    sepsis_id_index = pd.Index(sepsis_with_vitals)
    vital_chunks = []
    for chunk in pd.read_csv(path / "vitalPeriodic.csv", chunksize=500_000):
        kept = chunk[chunk["patientunitstayid"].isin(sepsis_id_index)]
        if not kept.empty:
            float_cols = kept.select_dtypes("float64").columns
            vital_chunks.append(kept.astype({c: "float32" for c in float_cols}))

    df_sepsis_vitals = pd.concat(vital_chunks, ignore_index=True)
    del vital_chunks

    # Sort each patient's records by time (oldest to newest)
    df_sepsis_vitals = df_sepsis_vitals.sort_values(
        ["patientunitstayid", "observationoffset"], ignore_index=True
    )

    if p:
        print(f"df_sepsis_vitals rows: {len(df_sepsis_vitals):,}")
        print(f"df_sepsis_vitals patients: {df_sepsis_vitals['patientunitstayid'].nunique():,}")
        print(df_sepsis_vitals[["patientunitstayid", "observationoffset"]].head())
        print(df_sepsis_vitals.loc[100:199, ["patientunitstayid", "observationoffset"]])

        pid = df_sepsis_vitals["patientunitstayid"]
        same_patient = pid.eq(pid.shift())
        is_sorted = not (same_patient & (df_sepsis_vitals["observationoffset"].diff() < 0)).any()
        print(f"offsets sorted within every patient: {is_sorted}")
        print(f"memory usage: {df_sepsis_vitals.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    # Drop low-coverage columns; temperature is replaced by nurseCharting
    df_sepsis_vitals = df_sepsis_vitals.drop(columns=DROP_COLS)
    n_vital_rows = len(df_sepsis_vitals)

    temp_df, gcs_df, temp_info = _load_nurse_charting(path, sepsis_id_index)
    extra = [temp_df, gcs_df, _load_nibp(path, sepsis_id_index), _load_urine(path, sepsis_id_index)]
    ext = reduce(lambda a, b: a.merge(b, on=KEYS, how="outer"), extra)
    ext = ext.astype({c: "float32" for c in ext.columns if c not in KEYS})
    del temp_df, gcs_df, extra

    # Keep each source's own timestamps; time binning happens in data_preprocessing
    df_sepsis_vitals = df_sepsis_vitals.merge(ext, on=KEYS, how="outer")
    del ext
    df_sepsis_vitals = df_sepsis_vitals.sort_values(KEYS, ignore_index=True)

    if p:
        n_patients = len(sepsis_with_vitals)
        print(f"temperature: {temp_info['converted']:,} converted from F, "
              f"{temp_info['dropped']:,} dropped (non-numeric or outside 25-45 C)")
        for col in ["temperature", "gcs", "nibp_systolic", "nibp_diastolic", "nibp_mean", "urine_output"]:
            has = df_sepsis_vitals[col].notna()
            n_pt = df_sepsis_vitals.loc[has, "patientunitstayid"].nunique()
            print(f"{col}: {int(has.sum()):,} records, {n_pt:,} patients ({n_pt / n_patients:.1%})")
        print(f"rows: {n_vital_rows:,} -> {len(df_sepsis_vitals):,}")
        print(f"heartrate records: {int(df_sepsis_vitals['heartrate'].notna().sum()):,}")

        pid = df_sepsis_vitals["patientunitstayid"]
        same_patient = pid.eq(pid.shift())
        is_sorted = not (same_patient & (df_sepsis_vitals["observationoffset"].diff() < 0)).any()
        print(f"offsets sorted within every patient: {is_sorted}")
        print(f"columns: {list(df_sepsis_vitals.columns)}")
        print(f"memory usage: {df_sepsis_vitals.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    return df_sepsis_vitals
