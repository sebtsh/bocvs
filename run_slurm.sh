#!/bin/sh
#SBATCH --job-name=cspbo
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sebastian.tay@u.nus.edu
#SBATCH --partition=long
#SBATCH --cpus-per-task=64
#SBATCH --time=4320
#SBATCH --exclude=xgpd6,xgpd7,xgpd9,amdgpu2

srun ./slurm_inner.sh
