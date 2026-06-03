#!/bin/bash
#SBATCH --job-name=datacomp_download
#SBATCH --output=datacomp_%j.log
#SBATCH --error=datacomp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gpus=4

set -e  # stop on any error

echo "activating environment"
TARBALL="$HOME/datacomp312.tar.gz"
ENV_DIR="/scratch/akash/datacomp312"

mkdir -p "$ENV_DIR"
echo "Extracting tarball from home..."
tar -xzf "$TARBALL" -C "$ENV_DIR"
source "$ENV_DIR/bin/activate"
echo "environment activated"

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# ---- Change these paths as needed ----
INPUT_DIR="/scratch/akash/00000000_tar/"
OUTPUT_PARQUET="/scratch/akash/00000000"
MODEL_DIR="/scratch/akash/models/"
BATCH_SIZE=15
# ---------------------------------------

echo "Starting gemma_cap.py on $(hostname) with $SLURM_GPUS_ON_NODE GPUs"
echo "Input:  $INPUT_DIR"

echo "paligemma caption generation started"
python paligemma_cap.py \
    --input_dir "$INPUT_DIR" \
    --output_parquet "$OUTPUT_PARQUET" \
    --model_dir "$MODEL_DIR" \
    --batch_size "$BATCH_SIZE"
echo "completed paligemma caption generation"

echo "moving parquet"
cp /scratch/akash/00000000..parquet /home2/akash.manna/ws/datacomp/data_curation/prompt_gen/
echo "parquet file moved"

echo "Done."
