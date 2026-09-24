#!/usr/bin/env bash
#SBATCH --job-name=horizon-preprocess
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/slurm/home/%u/output/%j/terminal.out
#SBATCH --error=/slurm/home/%u/output/%j/terminal.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/install_env.sh
python -m training.scripts.preprocess --input data/raw --output data/processed
