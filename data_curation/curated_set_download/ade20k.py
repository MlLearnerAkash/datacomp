"""
Download ADE20K dataset.

The full ADE20K dataset (25k train + 2k val images with full annotations)
requires registration at: https://ade20k.csail.mit.edu/request_data

This script downloads the ADE20K scene parsing benchmark via HuggingFace
(scene_parse_150), which is the most common variant used in semantic
segmentation research.
"""

import os
from huggingface_hub import snapshot_download

DATASET_ID = "scene_parse_150"
DOWNLOAD_DIR = "/scratch/ade20k"


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print("Downloading ADE20K scene parsing benchmark from HuggingFace...")
    print(f"Target: {DOWNLOAD_DIR}")

    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=DOWNLOAD_DIR,
        resume_download=True,
    )

    # Also download the consistency set directly
    consistency_url = "https://ade20k.csail.mit.edu/ADE20K_2017_05_30_consistency.zip"
    consistency_dest = os.path.join(DOWNLOAD_DIR, "ADE20K_2017_05_30_consistency.zip")
    if not os.path.exists(consistency_dest):
        print(f"\nDownloading consistency set: {consistency_url}")
        os.system(f'wget -t0 -c -O "{consistency_dest}" "{consistency_url}"')

    print("\nDone.")
    print(f"Data location: {DOWNLOAD_DIR}")
    print("\nNOTE: For the FULL ADE20K dataset (25k training images with full")
    print("object+part annotations), register at:")
    print("  https://ade20k.csail.mit.edu/request_data")


if __name__ == "__main__":
    main()
