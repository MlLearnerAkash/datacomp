"""
Unified script to download all curated datasets with wandb progress tracking.

Usage:
    python download_all.py --data-dir /scratch/akash/data

Datasets downloaded:
    - Pascal VOC 2007 & 2012 (train sets)
    - NYU v2 Depth (train shards)
    - ADE20K (scene parsing benchmark)
    - UnED (Universal Embeddings — direct downloads + manual instructions)
"""

import argparse
import os
import shutil
import subprocess
import time
import sys
from contextlib import contextmanager

import wandb
from huggingface_hub import hf_hub_download, snapshot_download

os.environ["WANDB_API_KEY"]= "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"
# ═══════════════════════════════════════════════════════════════════
#  Utility helpers
# ═══════════════════════════════════════════════════════════════════

def run_cmd(cmd: str) -> int:
    """Run a shell command, streaming output, return exit code."""
    proc = subprocess.run(cmd, shell=True, executable="/bin/bash")
    return proc.returncode


def dir_size_gb(path: str) -> float:
    """Return total size of all files under *path* in GB, or 0 if absent."""
    if not os.path.exists(path):
        return 0.0
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024**3)


@contextmanager
def wandb_step(name: str):
    """Log start/end of a dataset-download step to wandb."""
    t0 = time.time()
    wandb.log({f"{name}/status": "started"})
    try:
        yield
        elapsed = time.time() - t0
        size = dir_size_gb(os.path.join(DATA_DIR, name))
        wandb.log({
            f"{name}/status": "done",
            f"{name}/elapsed_min": round(elapsed / 60, 2),
            f"{name}/size_gb": round(size, 2),
        })
        print(f"[{name}] ✓ finished in {elapsed/60:.1f} min  ({size:.1f} GB)")
    except Exception as e:
        wandb.log({f"{name}/status": "failed", f"{name}/error": str(e)})
        print(f"[{name}] ✗ FAILED: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════
#  Individual dataset downloaders
# ═══════════════════════════════════════════════════════════════════

def download_pascal_voc(data_dir: str):
    """Download Pascal VOC 2007 & 2012 training sets."""
    voc_dir = os.path.join(data_dir, "VOC")
    os.makedirs(voc_dir, exist_ok=True)

    years = ["2007", "2012"]
    tar_files = {
        "2007": "VOCtrainval_06-Nov-2007",
        "2012": "VOCtrainval_11-May-2012",
    }

    for year in years:
        tar_name = tar_files[year]
        tar_path = os.path.join(voc_dir, tar_name + ".tar")
        url = f"http://host.robots.ox.ac.uk/pascal/VOC/voc{year}/{tar_name}.tar"

        if not os.path.isfile(tar_path):
            wandb.log({f"VOC/{year}_download": "started"})
            ret = run_cmd(f'wget -t0 -c -P "{voc_dir}" "{url}"')
            if ret != 0:
                raise RuntimeError(f"wget failed for VOC {year}")
            wandb.log({f"VOC/{year}_download": "done"})

        wandb.log({f"VOC/{year}_extract": "started"})
        ret = run_cmd(f'tar -xf "{tar_path}" -C "{voc_dir}"')
        if ret != 0:
            raise RuntimeError(f"tar extract failed for VOC {year}")
        wandb.log({f"VOC/{year}_extract": "done"})

    # count training images
    for year in years:
        train_txt = os.path.join(voc_dir, "VOCdevkit", f"VOC{year}", "ImageSets", "Main", "train.txt")
        with open(train_txt) as f:
            n = len(f.read().splitlines())
        wandb.log({f"VOC/{year}_train_images": n})
        print(f"  VOC {year} train images: {n}")


def download_nyu_depth(data_dir: str):
    """Download NYU v2 Depth training shards from HuggingFace."""
    repo = "sayakpaul/nyu_depth_v2"
    dest = os.path.join(data_dir, "nyu_depth_v2")
    os.makedirs(dest, exist_ok=True)

    num_shards = 12
    for i in range(num_shards):
        filename = f"data/train-{i:06d}.tar"
        wandb.log({"NYU/shard": i, "NYU/file": filename})
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=filename,
            local_dir=dest,
            resume_download=True,
        )
        print(f"  NYU v2: downloaded shard {i+1}/{num_shards}")

    wandb.log({"NYU/num_shards": num_shards})


def download_ade20k(data_dir: str):
    """Download ADE20K scene-parsing benchmark from HuggingFace."""
    repo = "scene_parse_150"
    dest = os.path.join(data_dir, "ade20k")

    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=dest,
        resume_download=True,
    )

    # consistency set
    url = "https://ade20k.csail.mit.edu/ADE20K_2017_05_30_consistency.zip"
    dest_zip = os.path.join(dest, "ADE20K_2017_05_30_consistency.zip")
    if not os.path.exists(dest_zip):
        run_cmd(f'wget -t0 -c -O "{dest_zip}" "{url}"')


def download_uned(data_dir: str):
    """Download UnED direct components; print manual-download instructions."""
    base = os.path.join(data_dir, "uned")
    os.makedirs(base, exist_ok=True)

    direct = [
        ("inat2018/train_val2018.tar.gz",
         "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz"),
        ("rp2k/rp2k_dataset.zip",
         "https://blob-nips2020-rp2k-dataset.obs.cn-east-3.myhuaweicloud.com/rp2k_dataset.zip"),
        ("info_files.zip",
         "https://cmp.felk.cvut.cz/univ_emb/info_files.zip"),
    ]

    for rel_path, url in direct:
        dest_file = os.path.join(base, rel_path)
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        if not os.path.exists(dest_file):
            wandb.log({"UnED/download": rel_path})
            ret = run_cmd(f'wget -t0 -c -O "{dest_file}" "{url}"')
            if ret != 0:
                raise RuntimeError(f"wget failed for UnED: {rel_path}")
            print(f"  UnED: downloaded {rel_path}")
        else:
            print(f"  UnED: already exists {rel_path}")

    # manual downloads
    manual = {
        "food2k":   "http://123.57.42.89/FoodProject.html (web form)",
        "cars196":  "https://www.kaggle.com/datasets/rvnrvn1/cars196/ (Kaggle CLI)",
        "sop":      "https://cvgl.stanford.edu/projects/lifted_struct/",
        "inshop":   "gdown '0B7EVK8r0v71pS2YxRE1QTFZzekU' (Google Drive)",
        "met":      "https://cmp.felk.cvut.cz/met",
        "gldv2":    "https://github.com/cvdfoundation/google-landmark",
    }
    for name, src in manual.items():
        os.makedirs(os.path.join(base, name), exist_ok=True)
        wandb.log({f"UnED/manual_{name}": src})
        print(f"  UnED [{name}]: MANUAL — {src}")


# ═══════════════════════════════════════════════════════════════════
#  Main orchestration
# ═══════════════════════════════════════════════════════════════════

DATA_DIR = ""   # set by main()


def main():
    global DATA_DIR

    parser = argparse.ArgumentParser(description="Download all curated datasets")
    parser.add_argument(
        "--data-dir", required=True,
        help="Parent directory where all datasets will be downloaded (e.g. /scratch/akash/data)",
    )
    parser.add_argument(
        "--wandb-project", default="dataset-download",
        help="wandb project name (default: dataset-download)",
    )
    parser.add_argument(
        "--wandb-entity", default=None,
        help="wandb entity/team name",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=["voc", "nyu", "ade20k", "uned"],
        help="Datasets to skip",
    )
    args = parser.parse_args()

    DATA_DIR = os.path.abspath(args.data_dir)
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- wandb init ---
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "data_dir": DATA_DIR,
            "datasets": ["voc", "nyu", "ade20k", "uned"],
        },
        tags=["dataset-download", "curation"],
        job_type="download",
    )
    wandb.log({"data_dir": DATA_DIR})

    # --- datasets to download ---
    datasets = [
        ("VOC",     download_pascal_voc),
        ("NYU",     download_nyu_depth),
        ("ADE20K",  download_ade20k),
        ("UnED",    download_uned),
    ]

    results = {}
    t_start = time.time()

    for name, fn in datasets:
        if name.lower() in args.skip:
            print(f"\n[{name}] SKIPPED (--skip)")
            wandb.log({f"{name}/status": "skipped"})
            continue

        print(f"\n{'='*60}")
        print(f"[{name}] starting ...")
        print(f"{'='*60}")
        try:
            with wandb_step(name):
                fn(DATA_DIR)
            results[name] = "done"
        except Exception:
            results[name] = "failed"
            # continue with other datasets instead of aborting
            continue

    # --- final summary ---
    total_elapsed = (time.time() - t_start) / 60
    total_size = dir_size_gb(DATA_DIR)
    wandb.log({
        "summary/total_elapsed_min": round(total_elapsed, 2),
        "summary/total_size_gb": round(total_size, 2),
        "summary/results": str(results),
    })

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")
    print(f"  Data dir    : {DATA_DIR}")
    print(f"  Total size  : {total_size:.1f} GB")
    print(f"  Total time  : {total_elapsed:.1f} min")
    print(f"  Results     : {results}")
    print(f"{'='*60}")

    wandb.finish()


if __name__ == "__main__":
    main()
