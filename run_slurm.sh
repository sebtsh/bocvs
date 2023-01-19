#!/bin/sh
#SBATCH --job-name=cvpbo
#SBATCH --partition=long
#SBATCH --cpus-per-task=64
#SBATCH --time=4320

srun ./slurm_inner.sh
