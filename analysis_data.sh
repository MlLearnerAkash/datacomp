#!/bin/bash
#SBATCH --job-name=datacomp_download
#SBATCH --output=datacomp_road_data_analysis_%j.log
#SBATCH --error=datacomp_road_data_analysis_%j.err
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

echo "Analysis starting..."
python analysis_data.py
echo "anaysis done."