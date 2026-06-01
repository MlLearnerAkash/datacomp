#ref: https://gist.github.com/turicas/b36fb1876b40888d92f5b2eefa2e9779
import os
import sys
import time
import timm
import wandb
from pathlib import Path

import timm
import torch
from PIL import Image

class FeatureExtractor:
    """Extract embeddings from images using timm's Dinov2 models"""
    model_names = (
        # "timm/vit_small_patch14_dinov2.lvd142m",
        # "timm/vit_base_patch14_dinov2.lvd142m",
        # "timm/vit_large_patch14_dinov2.lvd142m",
        "timm/vit_giant_patch14_dinov2.lvd142m",
        "vit_huge_patch14_224.orig_in21k" # https://huggingface.co/timm/vit_huge_patch14_224.orig_in21k
        )

    def __init__(self, model_name, device="cpu"):

        if model_name not in self.model_names:
            raise ValueError(f"Unknown model name: {repr(model_name)}")
        self.device = device
        self.model = timm.create_model(model_name, pretrained=True, 
                                       num_classes=0) # remove classifier nn.Linear
        self.model= self.model.to(device)
        self.model.eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def load_normalize(self, image: str | Path | Image.Image):
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        return self.transforms(img).to(self.device).unsqueeze(0)

    @torch.no_grad
    def extract(self, image: str | Path | Image.Image):
        input_tensor = self.load_normalize(image)
        output = self.model(input_tensor).squeeze(0)
        normalized = torch.nn.functional.normalize(output, dim=-1)
        return normalized

    @torch.no_grad
    def extract_batch(self, image_filenames):
        """Run a batch embedding extraction using `image_filenames`"""
        input_tensor = torch.cat([self.load_normalize(image_filename) for image_filename in image_filenames])
        output = self.model(input_tensor)
        normalized = torch.nn.functional.normalize(output, dim=-1)
        return normalized

    @torch.no_grad
    def extract_many(self, image_filenames, batch_size=16):
        """Extract embeddings from `image_filenames` in batches, using `batch_size`"""
        batch = []
        for filename in image_filenames:
            batch.append(filename)
            if len(batch) == batch_size:
                yield from zip(batch, self.extract_batch(batch))
                batch = []
        if batch:
            yield from zip(batch, self.extract_batch(batch))

if __name__ == "__main__":
    import argparse
    import traceback

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--device", "-d", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_name", "-m", type= str)
    parser.add_argument("model_size", choices=["small", "base", "large", "giant"], default= "base", help="Size of Dinov2 model")
    args = parser.parse_args()
    device = args.device
    batch_size = args.batch_size
    model_size = args.model_size
    model_name = args.model_name if args.model_name else f"timm/vit_{model_size}_patch14_dinov2.lvd142m"
    extractor = FeatureExtractor(model_name, device=device)
    
    #TODO: pass filtered images
    filenames= None
    for image_filename, embedding in extractor.extract_batch(filenames, batch_size= batch_size):
        pass



