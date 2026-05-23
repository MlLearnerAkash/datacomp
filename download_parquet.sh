#!/bin/bash
#SBATCH --job-name=datacomp_download
#SBATCH --output=datacomp_%j.log
#SBATCH --error=datacomp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --nodelist=gnode027

# Initialize Conda
source /home2/akash.manna/miniconda3/etc/profile.d/conda.sh
# Activate the specific environment
conda activate /home2/akash.manna/miniconda3/envs/datacomp
echo "Environment activated"

echo "starting parquets download..."
# Execute the download script
python download_upstream.py \
    --scale datacomp_1b \
    --data_dir /scratch/akash/ \
    --skip_bbox_blurring \
    --skip_shards \
    --resume_metadata
echo "parquets downloaded."

# Activate conda environment /home2/akash.manna/miniconda3/envs/datacomp
# python download_upstream.py --scale datacomp_1b  --data_dir /scratch/akash/ --skip_bbox_blurring --skip_shards
