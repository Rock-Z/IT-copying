#! /bin/bash

#SBATCH --nodes=1
#SBATCH --partition gpu
#SBATCH --cpus-per-gpu=2
#SBATCH --gpus=2
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB
#SBATCH --mail-user=enyan.zhang@yale.edu

uv run src/generate.py

exit 0