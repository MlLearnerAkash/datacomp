#!/bin/bash
#SBATCH --job-name=curated_set_download
#SBATCH --output=curated_set_download_%j.log
#SBATCH --error=curated_set_download_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --nodelist=gnode042

# Initialize Conda
source /home2/akash.manna/miniconda3/etc/profile.d/conda.sh
# Activate the specific environment
conda activate /home2/akash.manna/miniconda3/envs/datacomp
echo "Environment activated"

echo "starting download..."
python download_all.py --data-dir /scratch/akash/ --skip ade20k
echo "all downloaded."