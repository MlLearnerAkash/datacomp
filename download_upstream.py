import argparse
import os
import re
import shutil
from pathlib import Path

import img2dataset
from cloudpathlib import CloudPath
from huggingface_hub import list_repo_files, hf_hub_download

from scale_configs import available_scales

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"


def path_or_cloudpath(s):
    if re.match(r"^\w+://", s):
        return CloudPath(s)
    return Path(s)


def cleanup_dir(path):
    assert isinstance(path, Path) or isinstance(path, CloudPath)
    if isinstance(path, Path):
        shutil.rmtree(path)
    else:
        path.rmtree()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scale",
        type=str,
        required=False,
        choices=available_scales(simple_names=True)[1:] + ["datacomp_1b"],
        default="small",
        help="Competition scale.",
    )
    parser.add_argument(
        "--data_dir",
        type=path_or_cloudpath,
        required=True,
        help="Path to directory where the data (webdataset shards) will be stored.",
    )
    parser.add_argument(
        "--metadata_dir",
        type=path_or_cloudpath,
        default=None,
        help="Path to directory where the metadata (parquet files) is stored. If not set, infer from data_dir.",
    )
    parser.add_argument(
        "--skip_bbox_blurring",
        help="If true, skip bounding box blurring on images while downloading.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--processes_count",
        type=int,
        required=False,
        default=16,
        help="Number of processes for download.",
    )
    parser.add_argument(
        "--thread_count",
        type=int,
        required=False,
        default=128,
        help="Number of threads for download.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        required=False,
        default=512,
        help="Size images need to be downloaded to.",
    )
    parser.add_argument(
        "--resize_mode",
        type=str,
        required=False,
        choices=["no", "border", "keep_ratio", "keep_ratio_largest", "center_crop"],
        default="keep_ratio_largest",
        help="Resizing mode used by img2dataset when downloading images.",
    )
    parser.add_argument(
        "--no_resize_only_if_bigger",
        help="If true, do not resize only if images are bigger than target size.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--encode_format",
        type=str,
        required=False,
        choices=["png", "jpg", "webp"],
        default="jpg",
        help="Images encoding format.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        required=False,
        choices=["webdataset", "tfrecord", "parquet", "files"],
        default="webdataset",
        help="Output format used by img2dataset when downloading images.",
    )
    parser.add_argument(
        "--max_parquet_files",
        type=int,
        required=False,
        default=None,
        help="Maximum number of parquet files to download images from. "
        "If not set, all parquet files are used.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        required=False,
        default=2,
        help="Number of time a download should be retried (default 2)",
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        default=True,
        help="Whether to enable wandb logging (default False)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        required=False,
        default="datacomp",
        help="Name of W&B project used (default datacomp)",
    )
    parser.add_argument(
        "--skip_npz",
        action="store_true",
        default= False
    )
    parser.add_argument(
        "--skip_metadata",
        action="store_true",
        default= False
    )
    parser.add_argument(
        "--skip_shards",
        action="store_true",
        default= False
    )
    parser.add_argument(
        "--overwrite_metadata",
        action="store_true",
        default= False
    )
    

    args = parser.parse_args()

    hf_repo = (
        f"mlfoundations/datacomp_{args.scale}"
        if args.scale != "datacomp_1b"
        else "mlfoundations/datacomp_1b"
    )

    metadata_dir = args.metadata_dir
    if metadata_dir is None:
        metadata_dir = args.data_dir / "metadata"
    # Determine which parquet files already exist locally.
    existing_parquets = set()
    if metadata_dir.exists():
        for f in metadata_dir.glob("*.parquet"):
            if f.stat().st_size > 0:
                existing_parquets.add(f.name)

    should_download = args.overwrite_metadata or len(existing_parquets) == 0
    should_resume = getattr(args, "resume_metadata", False) and len(existing_parquets) > 0

    if not args.skip_metadata:
        if should_download or should_resume:
            if metadata_dir.exists() and args.overwrite_metadata:
                print(f"Cleaning up {metadata_dir}")
                shutil.rmtree(metadata_dir)
                metadata_dir.mkdir(parents=True)
            elif not metadata_dir.exists():
                metadata_dir.mkdir(parents=True)

            if should_resume and len(existing_parquets) > 0:
                # Only download parquet files that are not already present locally.
                print(f"Resuming: checking for missing parquet files among {len(existing_parquets)} existing...")
                remote_files = list_repo_files(hf_repo, repo_type="dataset")
                missing_parquets = [
                    f for f in remote_files
                    if f.endswith(".parquet") and Path(f).name not in existing_parquets
                ]
                if not missing_parquets:
                    print("All parquet files already present locally. Skipping metadata download.")
                else:
                    print(f"Downloading {len(missing_parquets)} missing parquet file(s)...")
                    for f in missing_parquets:
                        hf_hub_download(
                            repo_id=hf_repo,
                            filename=f,
                            local_dir=metadata_dir,
                            local_dir_use_symlinks=False,
                            repo_type="dataset",
                            resume_download=True,
                        )


    if not args.skip_npz:
        cache_dir = metadata_dir.parent / "hf"
        print("\nDownloading npz files")
        remote_files = list_repo_files(hf_repo, repo_type="dataset")
        npz_files = [f for f in remote_files if f.endswith(".npz")]
        if not npz_files:
            print("No npz files found.")
        else:
            f = npz_files[0] #NOTE: downlod signle file
            print(f"Downloading {f}")
            hf_hub_download(
                repo_id=hf_repo,
                filename=f,
                local_dir=metadata_dir,
                cache_dir=cache_dir,
                local_dir_use_symlinks=False,
                repo_type="dataset",
                resume_download=True,
            )
        cleanup_dir(cache_dir)

    # Flatten directory structure in case of xlarge
    if args.scale == "xlarge":
        filenames = list(metadata_dir.rglob("*.parquet")) + list(
            metadata_dir.rglob("*.npz")
        )
        for filename in filenames:
            basename = filename.name
            filename.replace(metadata_dir / basename)

        empty_dirs = list(metadata_dir.glob("part_*"))
        for empty_dir in empty_dirs:
            empty_dir.rmdir()

            print("Done downloading metadata.")
        else:
            print(
                f"Skipping download of metadata because {metadata_dir} exists. Use --overwrite_metadata to force re-downloading."
            )
    else:
        print("Skipping metadata download (--skip_metadata set).")

    if not args.skip_shards:
        # Download images.
        shard_dir = args.data_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading images to {shard_dir}")

        bbox_col = None if args.skip_bbox_blurring else "face_bboxes"

        # Determine which metadata directory to use for img2dataset.
        # If max_parquet_files is set, create a temp dir with only that many parquet files.
        url_list_dir = metadata_dir
        temp_metadata_dir = None
        skip_image_download = False
        if args.max_parquet_files is not None:
            import tempfile
            parquet_files = sorted(metadata_dir.glob("*.parquet"))
            if not parquet_files:
                print("No parquet files found in metadata directory. Did you forget to download metadata?")
                skip_image_download = True
            else:
                selected = parquet_files[: args.max_parquet_files]
                print(
                    f"Limiting to {len(selected)} of {len(parquet_files)} parquet files "
                    f"(max_parquet_files={args.max_parquet_files})."
                )
                temp_metadata_dir = Path(tempfile.mkdtemp(prefix="datacomp_metadata_"))
                for pf in selected:
                    shutil.copy(pf, temp_metadata_dir / pf.name)
                url_list_dir = temp_metadata_dir
        elif not metadata_dir.exists() or not any(metadata_dir.glob("*.parquet")):
            print("No parquet files found in metadata directory. Did you forget to download metadata?")
            skip_image_download = True

        if not skip_image_download:
            img2dataset.download(
                url_list=str(url_list_dir),
                image_size=args.image_size,
                output_folder=str(shard_dir),
                processes_count=args.processes_count,
                thread_count=args.thread_count,
                resize_mode=args.resize_mode,
                resize_only_if_bigger=not args.no_resize_only_if_bigger,
                encode_format=args.encode_format,
                output_format=args.output_format,
                input_format="parquet",
                url_col="url",
                caption_col="text",
                bbox_col=bbox_col,
                save_additional_columns=["uid"],
                number_sample_per_shard=10000,
                oom_shard_count=8,
                retries=args.retries,
                enable_wandb=args.enable_wandb,
                wandb_project=args.wandb_project,
            )

            # Clean up temporary metadata directory if one was created.
            if temp_metadata_dir is not None:
                cleanup_dir(temp_metadata_dir)
    else:
        print(f"Skipping image data download.")

    print("Done!")
