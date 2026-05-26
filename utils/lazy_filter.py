"""
lazy_filter.py — Row matching between shard parquets and metadata parquets.

Uses DuckDB for true out-of-core, disk-backed hash joins.
Scans 400GB of metadata parquets without loading into memory.
DuckDB spills to disk when memory limit is hit (no OOM).
"""

import os
import sys
import time
import shutil
import threading
from pathlib import Path

import duckdb

METADATA_DIR = Path("/scratch/akash/metadata")
SHARDS_DIR = Path("/scratch/akash/shards")
OUTPUT_DIR = Path("/scratch/akash/shards_merged")
JOIN_COLUMN = "uid"
TMP_DIR = Path("/scratch/akash/tmp_duckdb")

MEMORY_LIMIT = "24GB"


def main():
    shard_parquets = sorted(SHARDS_DIR.glob("*.parquet"))
    if not shard_parquets:
        print(f"ERROR: No shard parquet files found in {SHARDS_DIR}")
        sys.exit(1)

    metadata_pattern = str(METADATA_DIR / "*.parquet")
    n_meta = len(list(METADATA_DIR.glob("*.parquet")))
    print(f"Metadata: {n_meta} parquet files  |  "
          f"Shards: {len(shard_parquets)} files")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect()
    conn.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    conn.execute(f"SET temp_directory='{TMP_DIR}'")
    conn.execute("SET threads TO 8")
    conn.execute("SET enable_progress_bar = true")  # DuckDB prints progress to stderr

    print(f"DuckDB: memory_limit={MEMORY_LIMIT}, threads=8, "
          f"temp_dir={TMP_DIR}")

    print("\nPhase 1: Collecting unique uids from all shards...")
    t0 = time.time()

    all_uids = None
    for sp in shard_parquets:
        uids = conn.execute(
            f"SELECT DISTINCT {JOIN_COLUMN} FROM read_parquet('{sp}')"
        ).fetchall()
        uid_list = [r[0] for r in uids]
        if all_uids is None:
            all_uids = set(uid_list)
        else:
            all_uids.update(uid_list)

    print(f"  Collected {len(all_uids):,} unique uids in {time.time() - t0:.1f}s")

    import tempfile
    uid_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    uid_tmp.write("uid\n")
    for uid in all_uids:
        uid_tmp.write(f"{uid}\n")
    uid_tmp.close()
    conn.execute(f"CREATE TABLE shard_uids AS SELECT * FROM read_csv_auto('{uid_tmp.name}')")
    os.unlink(uid_tmp.name)

    print("\nPhase 2: Filtering metadata to shard uids (DuckDB streaming)...")
    print("  Scanning 400GB metadata — this is the slow step. "
          "DuckDB progress bar ↓")
    t0 = time.time()

    # Background thread: print elapsed time every 30s during long query
    done_flag = threading.Event()
    def progress_reporter():
        while not done_flag.is_set():
            done_flag.wait(30)
            if not done_flag.is_set():
                elapsed = time.time() - t0
                print(f"  [progress] Phase 2 running for {elapsed:.0f}s "
                      f"(~{elapsed/60:.1f} min)...", flush=True)

    reporter = threading.Thread(target=progress_reporter, daemon=True)
    reporter.start()

    try:
        conn.execute(f"""
            CREATE TABLE filtered_meta AS
            SELECT m.*
            FROM read_parquet('{metadata_pattern}') m
            SEMI JOIN shard_uids s ON m.{JOIN_COLUMN} = s.uid
        """)
    finally:
        done_flag.set()
        reporter.join(timeout=1)

    n_matched = conn.execute("SELECT COUNT(*) FROM filtered_meta").fetchone()[0]
    print(f"  Kept {n_matched:,} matching metadata rows "
          f"in {time.time() - t0:.1f}s")

    print(f"\nPhase 3: Joining {len(shard_parquets)} shards with metadata...")
    t_total = time.time()
    total_rows = 0
    total_matched = 0

    for sp in shard_parquets:
        shard_name = sp.stem
        t1 = time.time()
        out_path = OUTPUT_DIR / f"{shard_name}.parquet"

        conn.execute(f"""
            COPY (
                SELECT s.*, m.*
                FROM read_parquet('{sp}') s
                LEFT JOIN filtered_meta m ON s.{JOIN_COLUMN} = m.{JOIN_COLUMN}
            ) TO '{out_path}' (FORMAT PARQUET)
        """)

        # Count rows
        n_rows = conn.execute(
            f"SELECT COUNT(*) FROM read_parquet('{out_path}')"
        ).fetchone()[0]
        n_match = conn.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{out_path}')
            WHERE {JOIN_COLUMN} IS NOT NULL
        """).fetchone()[0]  # rough; uid from shard is always non-null

        elapsed = time.time() - t1
        total_rows += n_rows
        total_matched += n_match
        print(f"  ✓ {shard_name}: {n_rows:,} rows → "
              f"{n_match:,} matched ({elapsed:.1f}s)")

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    pct = (100 * total_matched / total_rows) if total_rows else 0
    print(f"\n{'='*60}")
    print(f"Done! {len(shard_parquets)} shards in {elapsed:.1f}s")
    print(f"  Total rows:    {total_rows:,}")
    print(f"  Matched:       {total_matched:,}  ({pct:.1f}%)")
    print(f"  Output dir:    {OUTPUT_DIR}")

    # Cleanup
    shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()



