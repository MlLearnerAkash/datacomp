"""
Download all subdomains of the Universal Embeddings (UnED) dataset.
Direct downloads are automated; manual-download domains print instructions.

Subdomains and their sources:
  Food2k        – manual (web form)
  CARS196       – manual (Kaggle)
  SOP           – manual (Stanford page)
  InShop        – manual (Google Drive)
  iNaturalist   – direct  (tar.gz)
  Met           – manual (web page)
  GLDv2         – manual (GitHub repo)
  Rp2k          – direct  (zip)
  Ground-truth  – direct  (zip)
"""

import os

BASE_DIR = "/scratch/akash/uned"


def download_direct(url, dest_dir, filename=None):
    """Download a file via wget into dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    if filename is None:
        filename = os.path.basename(url)
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest):
        print(f"[SKIP] Already exists: {dest}")
        return dest
    print(f"[DOWNLOADING] {url} -> {dest}")
    os.system(f'wget -t0 -c -O "{dest}" "{url}"')
    return dest


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    # ---- Direct downloads (fully automated) ----

    print("=" * 60)
    print("DIRECT DOWNLOADS")
    print("=" * 60)

    # iNaturalist 2018
    download_direct(
        "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz",
        os.path.join(BASE_DIR, "inat2018"),
    )

    # Rp2k
    download_direct(
        "https://blob-nips2020-rp2k-dataset.obs.cn-east-3.myhuaweicloud.com/rp2k_dataset.zip",
        os.path.join(BASE_DIR, "rp2k"),
    )

    # Ground-truth splits (info_files.zip)
    download_direct(
        "https://cmp.felk.cvut.cz/univ_emb/info_files.zip",
        BASE_DIR,
    )

    # ---- Manual downloads (print instructions) ----

    print("\n" + "=" * 60)
    print("MANUAL DOWNLOADS — follow instructions below")
    print("=" * 60)

    manual = [
        (
            "Food2k",
            os.path.join(BASE_DIR, "food2k"),
            "Go to http://123.57.42.89/FoodProject.html and fill the form to request access.\n"
            "Place the downloaded files in this directory.",
        ),
        (
            "CARS196",
            os.path.join(BASE_DIR, "cars196"),
            "Download from Kaggle: https://www.kaggle.com/datasets/rvnrvn1/cars196/\n"
            "Use 'kaggle datasets download rvnrvn1/cars196' if you have the Kaggle CLI.\n"
            "Place the downloaded files in this directory.",
        ),
        (
            "SOP (Stanford Online Products)",
            os.path.join(BASE_DIR, "sop"),
            "Download from: https://cvgl.stanford.edu/projects/lifted_struct/\n"
            "Get 'Stanford_Online_Products.zip' and place it in this directory.",
        ),
        (
            "InShop",
            os.path.join(BASE_DIR, "inshop"),
            "Download from Google Drive:\n"
            "https://drive.google.com/file/d/0B7EVK8r0v71pS2YxRE1QTFZzekU/view\n"
            "Use gdown:  gdown '0B7EVK8r0v71pS2YxRE1QTFZzekU'\n"
            "Place the downloaded file in this directory.",
        ),
        (
            "Met Dataset",
            os.path.join(BASE_DIR, "met"),
            "Go to https://cmp.felk.cvut.cz/met and follow instructions.\n"
            "Place the downloaded files in this directory.",
        ),
        (
            "GLDv2 (Google Landmarks Dataset v2)",
            os.path.join(BASE_DIR, "gldv2"),
            "Follow instructions at: https://github.com/cvdfoundation/google-landmark\n"
            "Place the downloaded files in this directory.",
        ),
    ]

    for name, dest_dir, instructions in manual:
        os.makedirs(dest_dir, exist_ok=True)
        print(f"\n--- {name} ---")
        print(f"  Target dir: {dest_dir}")
        print(f"  {instructions.strip()}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"All data should be placed under: {BASE_DIR}/")
    print("Direct downloads: inat2018/, rp2k/, info_files.zip")
    print("Manual downloads:  food2k/, cars196/, sop/, inshop/, met/, gldv2/")


if __name__ == "__main__":
    main()
