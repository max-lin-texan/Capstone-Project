"""CP decomposition of the 0-18h median tensor (patient x variable x window).

Two blood-pressure representations are compared, since MAP is highly
correlated with systolic/diastolic and using all three would double-count BP:
    Group A: systolic + diastolic (no MAP)
    Group B: MAP (no systolic/diastolic)
For each group, ranks 1-10 are fitted with several random starts and all
factors, a fit summary and a fit-vs-rank plot are saved under results/.
This is done for two datasets: with GCS and without GCS (see DATASETS).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to file without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyttb as ttb

# ---- Paths ----
PROJECT_DIR = Path.home() / "Capstone-Project"
DATA_DIR = PROJECT_DIR / "data"
RESULTS_ROOT = PROJECT_DIR / "results"

# ---- Analysis settings ----
WINDOWS = ["0-6", "6-12", "12-18"]  # chronological order of the time mode
BASE_VARS = ["sao2", "heartrate", "respiration", "temperature", "gcs"]  # shared by both groups

def make_groups(base_vars):
    """Group A uses systolic + diastolic, group B uses MAP; both share base_vars."""
    return {
        "A": base_vars + ["nibp_systolic", "nibp_diastolic"],
        "B": base_vars + ["nibp_mean"],
    }

# Each dataset: (input csv, output directory name, variable groups)
DATASETS = [
    # 7,315 patients complete in all 3 windows, including GCS
    ("df_median_complete.csv", "cpd_median_complete", make_groups(BASE_VARS)),
    # 16,805 patients complete in all 3 windows when GCS is not required
    ("df_median_complete_nogcs.csv", "cpd_median_complete_nogcs",
     make_groups([v for v in BASE_VARS if v != "gcs"])),
]
GROUP_LABELS = {"A": "A: systolic + diastolic", "B": "B: MAP"}  # legend text in the plot
RANKS = range(1, 11)  # number of components to try
SEEDS = range(5)     # random starts per rank; cp_als is non-convex so results depend on the start

def build_tensor(df, variables):
    """Return (patient ids, z-scored array of shape patient x variable x window)."""
    # As plain strings "12-18" would sort before "6-12"; force chronological order
    df = df.assign(window=pd.Categorical(df["window"], categories=WINDOWS, ordered=True))
    df = df.sort_values(["patientunitstayid", "window"])
    ids = df["patientunitstayid"].unique()

    # Every patient has exactly one row per window, so rows can be reshaped directly:
    # rows are (patient, window) -> (patient, window, variable) -> (patient, variable, window)
    arr = df[variables].to_numpy(dtype=float).reshape(len(ids), len(WINDOWS), len(variables))
    arr = arr.transpose(0, 2, 1)

    # z-score each variable over all patients and windows, so units (bpm, °C, mmHg)
    # do not let large-valued variables dominate the decomposition
    mean = arr.mean(axis=(0, 2), keepdims=True)
    std = arr.std(axis=(0, 2), keepdims=True)
    return ids, (arr - mean) / std

def best_cp(tensor, rank):
    """Run cp_als from several random starts; keep the best fit.

    Returns ((ktensor, fit, seed) of the best run, list of fits of all runs).
    """
    best, fits = None, []
    for seed in SEEDS:
        np.random.seed(seed)  # cp_als draws its random initial factors from numpy's global RNG
        M, _, info = ttb.cp_als(tensor, rank, printitn=0)
        fits.append(info["fit"])  # fit = 1 - ||X - M|| / ||X||; 1 means perfect reconstruction
        if best is None or info["fit"] > best[1]:
            best = (M, info["fit"], seed)
    return best, fits

def save_factors(M, name, rank, ids, variables, results_dir):
    """Save the three factor matrices and the component weights of one model."""
    cols = [f"comp{i + 1}" for i in range(rank)]
    # One factor matrix per tensor mode, in the same order as the array axes
    patient, variable, time = M.factor_matrices

    # Patient scores: one row per patient, used later e.g. to relate components to outcomes
    df_patient = pd.DataFrame(patient, columns=cols)
    df_patient.insert(0, "patientunitstayid", ids)
    df_patient.to_csv(results_dir / f"{name}_rank{rank}_patient.csv", index=False)
    # Variable loadings: which variables make up each component
    pd.DataFrame(variable, index=variables, columns=cols).rename_axis("variable").to_csv(
        results_dir / f"{name}_rank{rank}_variable.csv")
    # Time loadings: how each component changes across the three windows
    pd.DataFrame(time, index=WINDOWS, columns=cols).rename_axis("window").to_csv(
        results_dir / f"{name}_rank{rank}_time.csv")
    # Weights (lambda): overall magnitude of each component, factors are unit-norm
    pd.DataFrame({"component": cols, "weight": M.weights}).to_csv(
        results_dir / f"{name}_rank{rank}_weights.csv", index=False)

def plot_fit(summary, results_dir):
    """Plot best fit vs rank for both groups; look for where the curve flattens."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, g in summary.groupby("group"):
        ax.plot(g["rank"], g["best_fit"], marker="o", label=GROUP_LABELS[name])
    ax.set_xlabel("Rank")
    ax.set_ylabel("Fit")
    ax.set_xticks(list(RANKS))
    ax.set_title(f"CP-ALS fit vs rank (best of {len(SEEDS)} starts)\n{results_dir.name}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "fit_vs_rank.png", dpi=150)
    plt.close(fig)

def run_cpd(data_file, results_name, groups):
    """Fit all groups x ranks on one dataset and save everything to results/<results_name>."""
    results_dir = RESULTS_ROOT / results_name
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_DIR / data_file)
    print(f"=== {data_file}: {df['patientunitstayid'].nunique():,} patients ===")

    rows = []  # one summary row per (group, rank)
    for name, variables in groups.items():
        ids, arr = build_tensor(df, variables)
        tensor = ttb.tensor(arr)  # cp_als needs a pyttb tensor, not a numpy array
        print(f"Group {name}: tensor {arr.shape} ({', '.join(variables)})")

        for rank in RANKS:
            (M, fit, seed), fits = best_cp(tensor, rank)
            save_factors(M, name, rank, ids, variables, results_dir)
            # fit_min/max/std across starts: a narrow range means the solution is stable
            rows.append({"group": name, "rank": rank, "best_fit": fit, "best_seed": seed,
                         "fit_min": min(fits), "fit_max": max(fits), "fit_std": np.std(fits)})
            print(f"  rank {rank}: best fit {fit:.4f} (seed {seed}), range {min(fits):.4f}-{max(fits):.4f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "fit_summary.csv", index=False)
    plot_fit(summary, results_dir)
    print(f"Results saved to {results_dir}")
    return summary

if __name__ == "__main__":
    for data_file, results_name, groups in DATASETS:
        run_cpd(data_file, results_name, groups)
