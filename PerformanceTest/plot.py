# PerformanceTest/plot.py
# -*- coding: utf-8 -*-
"""
Produce two combined figures from one benchmark CSV:
  1) Mean + Worst runtime vs n (linear, side-by-side)
  2) Mean + Worst runtime vs n (log scale, side-by-side)

Mean figures use outlier-filtered data (SEM error bars).
Worst figures use raw data (max runtime).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# =========================
# Config (edit here)
# =========================
CSV_FILE_NAME = "p_n2_20(100).csv"   # file under PerformanceTest/results/
FAMILY_PREFIX: Optional[str] = None

OUTLIER_MODE = "iqr"     # "iqr" | "factor" | "none"
IQR_K = 1.5
FACTOR_MULT = 5.0
MIN_GROUP = 3

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, CSV_FILE_NAME)

# =========================
# Data loading / cleaning
# =========================
def load_and_clean(csv_file: str, family_prefix: Optional[str]) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_file)
    total_before = len(df_raw)

    df = df_raw.copy()
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df["solver"]   = df["solver"].astype(str)
    df["n"] = pd.to_numeric(
        df["key"].astype(str).str.extract(r"_n(\d+)", expand=False),
        errors="coerce"
    )
    if family_prefix:
        df = df[df["key"].astype(str).str.startswith(family_prefix)]

    df = df.dropna(subset=["time_sec", "n", "solver"])
    print("=== Data cleaning ===")
    print(f"Loaded rows : {total_before}")
    print(f"Valid rows  : {len(df)}")
    print(f"Dropped rows: {total_before - len(df)}")
    if df.empty:
        raise ValueError("No valid rows after cleaning. Check CSV/FAMILY_PREFIX.")
    return df

# =========================
# Outlier filters
# =========================
def filter_outliers_series(s: pd.Series) -> pd.Series:
    if OUTLIER_MODE == "none":
        return s
    if OUTLIER_MODE == "factor":
        med = s.median()
        return s.where(s < FACTOR_MULT * med)
    # IQR mode
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - IQR_K * iqr, q3 + IQR_K * iqr
    return s.where((s >= low) & (s <= high))

def apply_outlier_filter_for_mean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    filtered = df.copy()
    filtered["time_sec"] = (
        filtered.groupby(["solver", "n"])["time_sec"].transform(filter_outliers_series)
    )
    filtered = filtered.dropna(subset=["time_sec"])
    # drop groups with too few samples
    grp_sizes = filtered.groupby(["solver", "n"])["time_sec"].size().reset_index(name="cnt")
    keep_pairs = set(map(tuple, grp_sizes[grp_sizes["cnt"] >= MIN_GROUP][["solver","n"]].values))
    filtered = filtered[[ (row.solver, row.n) in keep_pairs for row in filtered.itertuples() ]]

    after = len(filtered)
    print("=== Outlier filtering (mean) ===")
    print(f"Rows before: {before}, after: {after}, removed: {before - after}")
    return filtered

# =========================
# Aggregation
# =========================
def aggregate_mean_sem(df: pd.DataFrame):
    grp = df.groupby(["solver", "n"])["time_sec"]
    stats = grp.agg(["mean", "std", "count"]).reset_index()
    stats["sem"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
    mean_p = stats.pivot(index="n", columns="solver", values="mean").sort_index()
    sem_p  = stats.pivot(index="n", columns="solver", values="sem").reindex_like(mean_p).fillna(0.0)
    cnt_p  = stats.pivot(index="n", columns="solver", values="count").reindex_like(mean_p)
    return mean_p, sem_p, cnt_p

def aggregate_worst_raw(df: pd.DataFrame):
    grp = df.groupby(["solver", "n"])["time_sec"]
    stats = grp.agg(["max", "count"]).reset_index()
    max_p = stats.pivot(index="n", columns="solver", values="max").sort_index()
    cnt_p = stats.pivot(index="n", columns="solver", values="count").reindex_like(max_p)
    return max_p, cnt_p

# =========================
# Plot
# =========================
def legend_with_counts(ax, counts):
    handles, labels = ax.get_legend_handles_labels()
    totals = counts.sum(axis=0).to_dict()
    new_labels = [f"{lab} (N={int(totals.get(lab, 0))})" for lab in labels]
    ax.legend(handles, new_labels)

def plot_combined(mean_p, sem_p, cnt_mean, max_p, cnt_worst, base: str):
    # Linear scale figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    # Left: mean
    for s in mean_p.columns:
        axes[0].errorbar(mean_p.index, mean_p[s], yerr=sem_p[s], marker="o", capsize=4, label=s)
    axes[0].set_title("Mean runtime (linear, outliers filtered)")
    axes[0].set_xlabel("n (problem size)")
    axes[0].set_ylabel("Runtime (s)")
    axes[0].grid(True, ls="--", lw=0.5)
    legend_with_counts(axes[0], cnt_mean)
    # Right: worst
    for s in max_p.columns:
        axes[1].plot(max_p.index, max_p[s], marker="o", label=s)
    axes[1].set_title("Worst-case runtime (linear, raw)")
    axes[1].set_xlabel("n (problem size)")
    axes[1].set_ylabel("Runtime (s)")
    axes[1].grid(True, ls="--", lw=0.5)
    legend_with_counts(axes[1], cnt_worst)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{base}_linear.png"), dpi=200)
    plt.close()

    # Log scale figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    # Left: mean
    for s in mean_p.columns:
        axes[0].errorbar(mean_p.index, mean_p[s], yerr=sem_p[s], marker="o", capsize=4, label=s)
    axes[0].set_yscale("log")
    axes[0].set_title("Mean runtime (log, outliers filtered)")
    axes[0].set_xlabel("n (problem size)")
    axes[0].set_ylabel("Runtime (s, log)")
    axes[0].grid(True, which="both", ls="--", lw=0.5)
    legend_with_counts(axes[0], cnt_mean)
    # Right: worst
    for s in max_p.columns:
        axes[1].plot(max_p.index, max_p[s], marker="o", label=s)
    axes[1].set_yscale("log")
    axes[1].set_title("Worst-case runtime (log, raw)")
    axes[1].set_xlabel("n (problem size)")
    axes[1].set_ylabel("Runtime (s, log)")
    axes[1].grid(True, which="both", ls="--", lw=0.5)
    legend_with_counts(axes[1], cnt_worst)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{base}_log.png"), dpi=200)
    plt.close()

# =========================
# Main
# =========================
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df_raw = load_and_clean(CSV_PATH, FAMILY_PREFIX)
    df_mean = apply_outlier_filter_for_mean(df_raw)
    mean_p, sem_p, cnt_mean = aggregate_mean_sem(df_mean)
    max_p, cnt_worst = aggregate_worst_raw(df_raw)

    base = os.path.splitext(os.path.basename(CSV_PATH))[0]
    plot_combined(mean_p, sem_p, cnt_mean, max_p, cnt_worst, base)
    print("[OK] Saved combined figures in plots/")

if __name__ == "__main__":
    main()
