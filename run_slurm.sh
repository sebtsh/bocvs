#!/bin/sh
#SBATCH --job-name=cspbo
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sebastian.tay@u.nus.edu
#SBATCH --partition=long
#SBATCH --cpus-per-task=32
#SBATCH --time=4320

echo "obj_name: $1"
echo "acq_name: $2"
echo "eps_schedule_id: $3"

CUDA_VISIBLE_DEVICES=-1 srun python bigger_exp.py "$1" "$2" "$3"

# run with sbatch run_slurm.sh CONFIG_NAME OBJ_NAME SEED
