import os
from huggingface_hub import hf_hub_download

DATASET_ID = "sayakpaul/nyu_depth_v2"
DOWNLOAD_DIR = "/scratch/akash/nyu_depth_v2"
NUM_TRAIN_SHARDS = 12  # train-000000.tar through train-000011.tar


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for i in range(NUM_TRAIN_SHARDS):
        filename = f"data/train-{i:06d}.tar"
        local_path = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=DOWNLOAD_DIR,
            resume_download=True,
        )
        print(f"Downloaded: {filename} -> {local_path}")

    # list downloaded files
    data_dir = os.path.join(DOWNLOAD_DIR, "data")
    train_files = sorted(os.listdir(data_dir))
    print(f"\nDownloaded {len(train_files)} train shards to {data_dir}:")
    for f in train_files:
        size_gb = os.path.getsize(os.path.join(data_dir, f)) / (1024**3)
        print(f"  {f}  ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
