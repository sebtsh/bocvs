#!/bin/sh
#SBATCH --job-name=cspbo
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sebastian.tay@u.nus.edu
#SBATCH --partition=long
#SBATCH --cpus-per-task=32
#SBATCH --time=4320

srun ./slurm_inner.sh "$1"
