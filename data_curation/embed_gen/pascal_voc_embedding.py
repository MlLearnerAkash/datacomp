import os
import subprocess
import clip
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
VOC_ROOT = "/scratch/akash/VOC"
VOC_YEAR = "2012"
VOC_IMAGE_SET = "train"
TAR_URL = f"http://host.robots.ox.ac.uk/pascal/VOC/voc{VOC_YEAR}/VOCtrainval_11-May-2012.tar"
TAR_PATH = os.path.join(VOC_ROOT, "VOCtrainval_11-May-2012.tar")
VOC_DEVKIT = os.path.join(VOC_ROOT, "VOCdevkit")
VOC_YEAR_DIR = os.path.join(VOC_DEVKIT, f"VOC{VOC_YEAR}")


def ensure_voc_data():
    """Download & extract PASCAL VOC if not already present."""
    if os.path.isdir(VOC_YEAR_DIR):
        print(f"✓ VOC{VOC_YEAR} already exists at {VOC_YEAR_DIR}")
        return

    os.makedirs(VOC_ROOT, exist_ok=True)

    # Download with wget (faster, supports resume)
    if not os.path.isfile(TAR_PATH):
        print(f"Downloading VOC{VOC_YEAR} trainval (~2 GB) via wget ...")
        subprocess.run(
            ["wget", "-t", "0", "-c", "-P", VOC_ROOT, TAR_URL],
            check=True,
        )
    else:
        print(f"✓ Tar file already exists at {TAR_PATH}")

    # Extract
    print("Extracting tar file ...")
    subprocess.run(["tar", "-xf", TAR_PATH, "-C", VOC_ROOT], check=True)
    print("✓ Extraction complete")


# ── Model ───────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP model on {device} ...")
model, preprocess = clip.load('ViT-L/14', device)
print("✓ Model loaded")


class ImageOnlyDataset(Dataset):
    """Wrapper around VOCDetection that returns only the preprocessed image,
    so the DataLoader can batch them without collation issues."""
    def __init__(self, root, year, image_set, transform):
        self.dataset = VOCDetection(
            root=root, year=year, image_set=image_set,
            download=False, transform=transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image


def get_features(dataset, batch_size=100):
    """Extract CLIP image embeddings for the entire dataset."""
    all_features = []

    with torch.no_grad():
        for images in tqdm(DataLoader(dataset, batch_size=batch_size)):
            features = model.encode_image(images.to(device))
            all_features.append(features)

    return torch.cat(all_features).cpu().numpy()


# ── Main ────────────────────────────────────────────────────────────────────
ensure_voc_data()

print(f"Loading PASCAL VOC {VOC_YEAR} {VOC_IMAGE_SET} set ...")
train_dataset = ImageOnlyDataset(
    root=VOC_ROOT, year=VOC_YEAR, image_set=VOC_IMAGE_SET, transform=preprocess,
)
print(f"Number of training images: {len(train_dataset)}")

print("Extracting CLIP image embeddings ...")
train_features = get_features(train_dataset)

output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, f"pascal_voc_{VOC_YEAR}_{VOC_IMAGE_SET}_embeddings.npz")
np.savez(output_path, features=train_features)
print(f"✓ Saved embeddings to {output_path}")
print(f"  Shape: {train_features.shape}")
