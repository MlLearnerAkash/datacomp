import os
import sys
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl

METADATA_DIR = Path("/scratch/akash/metadata")
SHARDS_DIR = Path("/scratch/akash/shards")
OUTPUT_DIR = Path("/scratch/akash/shards_merged")
JOIN_COLUMN = "uid"
NUM_SHARD_WORKERS = 1

USE_GPU = False

def _detect_engine() -> str:
    """Return the polars engine string based on USE_GPU and availability."""
    if not USE_GPU:
        return "cpu"
    try:
        pl.LazyFrame({"x": [1]}).collect(engine="gpu")
        return "gpu"
    except Exception:
        print("WARNING: GPU engine not available, falling back to CPU.")
        return "cpu"

ENGINE = _detect_engine()
GPU_PREAMBLE = (
    pl.Config()
)
print(f"Using polars engine: {ENGINE}")

def get_metadata_paths(metadata_dir: Path) -> list:
    paths = sorted(metadata_dir.glob("*.parquet"))
    if not paths:
        print(f"ERROR: No parquet files found in {metadata_dir}")
        sys.exit(1)
    return [str(p) for p in paths]

def process_shard(shard_parquet: Path, meta_paths: list,
                  output_dir: Path, engine: str) -> dict:
    """
    Load one shard parquet, semi-join metadata, merge, sink result.
    All collect/sink calls use the configured engine (gpu or cpu-streaming).
    """
    meta_paths = meta_paths#[]
    shard_name = shard_parquet.stem
    t0 = time.time()

    try:
        shard_lf = pl.scan_parquet(str(shard_parquet))
        # Collect uid column only (lightweight, used for row count + lookup)
        shard_uids = shard_lf.select(pl.col(JOIN_COLUMN)).collect(engine=engine)
        shard_rows = len(shard_uids)
        if shard_rows == 0:
            return {
                "shard": shard_name, "rows": 0, "matched": 0,
                "unmatched": 0, "time": time.time() - t0,
                "status": "OK (empty)",
            }

        # Lazy scan metadata parquet(s)
        meta_lf = pl.scan_parquet(meta_paths, parallel="auto", low_memory=True)
        shard_uid_lf = shard_lf.select(pl.col(JOIN_COLUMN)).unique()
        relevant_meta = meta_lf.join(shard_uid_lf, on=JOIN_COLUMN, how="semi")

        # Right-join: keep all metadata matches, attach shard columns
        merged_lf = shard_lf.join(
            relevant_meta,
            on=JOIN_COLUMN,
            how="right",
            suffix="_metadata",
        )
        output_path = output_dir / f"{shard_name}.parquet"
        merged_lf.sink_parquet(str(output_path), engine=engine)

        # Read back to count matches
        result_df = pl.scan_parquet(str(output_path)).collect(engine=engine)
        matched = result_df.filter(
            pl.col(f"{JOIN_COLUMN}_metadata").is_not_null()
        ).height

        elapsed = time.time() - t0
        return {
            "shard": shard_name,
            "rows": shard_rows,
            "matched": matched,
            "unmatched": shard_rows - matched,
            "time": elapsed,
            "status": "OK",
        }
    except Exception as e:
        print("Encountered: ", e)
        return {
            "shard": shard_parquet.stem,
            "rows": 0, "matched": 0, "unmatched": 0,
            "time": time.time() - t0,
            "status": f"ERROR: {e}",
        }

def main():
    meta_paths = get_metadata_paths(METADATA_DIR)
    print(f"Found {len(meta_paths)} metadata parquet files")
    shard_parquets = sorted(SHARDS_DIR.glob("*.parquet"))
    if not shard_parquets:
        print(f"ERROR: No shard parquet files found in {SHARDS_DIR}")
        sys.exit(1)
    print(f"Found {len(shard_parquets)} shard parquet files.")

    if OUTPUT_DIR.exists():
        print(f"  Cleaning existing output dir: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    print(f"\nProcessing shards "
          f"({'parallel' if NUM_SHARD_WORKERS > 1 else 'sequential'}, "
          f"engine={ENGINE})...\n")
    t_total = time.time()

    results = []
    total_rows = 0
    total_matched = 0

    if NUM_SHARD_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=NUM_SHARD_WORKERS) as executor:
            futures = {
                executor.submit(process_shard, sp, meta_paths, OUTPUT_DIR,
                                ENGINE): sp
                for sp in shard_parquets
            }
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                total_rows += res["rows"]
                total_matched += res["matched"]
                print(f"total matched: ", total_matched)
    else:
        for sp in shard_parquets:
            res = process_shard(sp, meta_paths, OUTPUT_DIR, ENGINE)
            results.append(res)
            total_rows += res["rows"]
            total_matched += res["matched"]
            print(f"Total matched: ", total_matched)

    # Summary
    elapsed = time.time() - t_total
    total = total_rows
    pct = (100 * total_matched / total) if total else 0
    print(f"\n{'='*60}")
    print(f"Done! {len(results)} shards processed in {elapsed:.1f}s")
    print(f"  Total shard rows:    {total:,}")
    print(f"  Matched to metadata: {total_matched:,}  ({pct:.1f}%)")
    print(f"  Unmatched:           {total - total_matched:,}")
    print(f"  Output dir:          {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


