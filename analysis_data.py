import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CLASSES_FILE = os.path.join(os.path.dirname(__file__), "utils", "classes.txt")

def load_parquet_columns(filepath: str, columns: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(filepath, engine='pyarrow')
    if len(columns) == 0:
        columns = ["text", "clip_b32_similarity_score", "clip_l14_similarity_score"]
    else:
        columns= columns
    return df[columns]

def _load_classes() -> list[str]:
    """Load the list of class keywords from classes.txt."""
    with open(CLASSES_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]

def process_single_parquet(filepath: str, columns: list[str]) -> pd.DataFrame:
    """Load a parquet file, filter rows matching classes.txt, save as CSV, and return the DataFrame."""
    df = load_parquet_columns(filepath, columns)
    classes = _load_classes()
    # Build a case-insensitive regex pattern from all classes
    pattern = "(?i)"+ "|".join(rf"\b{re.escape(word)}\b" for word in classes)
    mask = df["text"].str.contains(pattern, case=False, na=False, regex=True)
    filtered = df[mask]

    # Save CSV with the same base name as the parquet file
    csv_path = os.path.splitext(filepath)[0] + ".csv"
    filtered.to_csv(csv_path, index=False)
    return filtered

def analyze_csvs(csv_dir: str, parquet_dir: str) -> dict:
    """Analyze all CSV files in csv_dir and print/summarize results.

    Args:
        csv_dir: Directory containing the saved CSV files.
        parquet_dir: Directory containing the original parquet files (for total-row percentage).

    Returns:
        dict with keys: 'pct_matched', 'length_dist', 'b32_dist', 'l14_dist'.
    """
    import glob

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))

    total_rows = 0
    matched_rows = 0
    all_filtered_parts = []

    for fpath in csv_files:
        if os.path.getsize(fpath) == 0:
            continue
        df = pd.read_csv(fpath, engine="python", on_bad_lines="skip")
        matched_rows += len(df)
        all_filtered_parts.append(df)

    # for fpath in parquet_files:
    #     pf = pd.read_parquet(fpath, engine="pyarrow", columns=["text"])
    #     total_rows += len(pf)

    # pct_matched = (matched_rows / total_rows * 100) if total_rows > 0 else 0.0
    # print(f"\n=== 1. Percentage of whole dataset matching classes ===")
    # print(f"    Matched: {matched_rows:,} / {total_rows:,}  ({pct_matched:.2f}%)")

    combined = pd.concat(all_filtered_parts, ignore_index=True)

    LENGTH_BINS = [0, 3, 5, 7, 10, 15, 20, float("inf")]
    length_labels = [
        f"[{LENGTH_BINS[i]}, {LENGTH_BINS[i+1]})"
        if LENGTH_BINS[i+1] != float("inf")
        else f">{LENGTH_BINS[i]}"
        for i in range(len(LENGTH_BINS) - 1)
    ]
    combined["word_count"] = combined["text"].fillna("").str.split().str.len()
    combined["length_bin"] = pd.cut(
        combined["word_count"], bins=LENGTH_BINS, labels=length_labels, right=False
    )
    length_dist = combined["length_bin"].value_counts().sort_index()

    print(f"\n=== 2. Caption length distribution (word count, exclusive) ===")
    for bin_label, count in length_dist.items():
        print(f"    {bin_label}: {count:,}")

    # Plot length distribution
    total = length_dist.values.sum()
    length_pct = length_dist.values / total * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(length_dist)), length_pct, width=0.5, color="darkorange", edgecolor="black")
    ax.set_xticks(range(len(length_dist)))
    ax.set_xticklabels(length_dist.index, rotation=30, ha="right")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Percentage of captions (%)")
    ax.set_title("Caption Length Distribution (word count)")
    plt.tight_layout()
    plt.savefig(os.path.join(csv_dir, "caption_length_distribution.png"), dpi=150)
    # plt.show()

    # --- 3. Score distributions (combined side-by-side plot) ---
    SCORE_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    score_labels = [
        f"({SCORE_BINS[i]:.1f}, {SCORE_BINS[i+1]:.1f}]" for i in range(len(SCORE_BINS) - 1)
    ]

    score_dists = {}
    for col, name in [
        ("clip_b32_similarity_score", "CLIP-B/32"),
        ("clip_l14_similarity_score", "CLIP-L/14"),
    ]:
        combined[f"{col}_bin"] = pd.cut(
            combined[col], bins=SCORE_BINS, labels=score_labels, right=True
        )
        dist = combined[f"{col}_bin"].value_counts().sort_index()
        score_dists[name] = dist

        print(f"\n=== 3. {name} score distribution ===")
        for bin_label, count in dist.items():
            print(f"    {bin_label}: {count:,}")

  

    n_bins = len(score_labels)
    x = np.arange(n_bins)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    b32_pct = score_dists["CLIP-B/32"].values / score_dists["CLIP-B/32"].values.sum() * 100
    l14_pct = score_dists["CLIP-L/14"].values / score_dists["CLIP-L/14"].values.sum() * 100
    bars_b32 = ax.bar(
        x - bar_width / 2, b32_pct,
        bar_width, label="CLIP-B/32", color="steelblue", edgecolor="black",
    )
    bars_l14 = ax.bar(
        x + bar_width / 2, l14_pct,
        bar_width, label="CLIP-L/14", color="darkorange", edgecolor="black",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(score_labels, rotation=30, ha="right")
    ax.set_xlabel("Score range")
    ax.set_ylabel("Percentage of captions (%)")
    ax.set_title("CLIP Score Distribution (B/32 vs L/14)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(csv_dir, "clip_score_distribution.png"), dpi=150)

    return {
        "length_dist": length_dist,
        "b32_dist": score_dists["CLIP-B/32"],
        "l14_dist": score_dists["CLIP-L/14"],
    }
if __name__ == "__main__":
    import glob

    PARQUET_DIR = "/scratch/akash/metadata"
    parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))

    print(f"Found {len(parquet_files)} parquet files in {PARQUET_DIR}")

    for i, fpath in enumerate(parquet_files):
        process_single_parquet(fpath, columns=[])
        # if i>1:
        #     break
    print("CSV file generated.")

    print("Starting CSV analysis" )
    analyze_csvs(csv_dir=PARQUET_DIR, parquet_dir=PARQUET_DIR)
    print("csv analysis done")