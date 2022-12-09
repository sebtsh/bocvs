#!/bin/sh
#SBATCH --job-name=cspbo
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sebastian.tay@u.nus.edu
#SBATCH --partition=long
#SBATCH --cpus-per-task=16
#SBATCH --time=4320

srun ./slurm_inner.sh "$1" "$2" "$3"
