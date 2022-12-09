#!/bin/sh
#SBATCH --job-name=msbo
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sebastian.tay@u.nus.edu
#SBATCH --partition=long
#SBATCH --ntasks=64
#SBATCH --time=4320

echo "obj_name: $1"
echo "acq_name: $2"
echo "eps_schedule_id: $3"

wrapper(){
  source ~/anaconda3/bin/activate
}
wrapper
conda activate cspbo

CUDA_VISIBLE_DEVICES=-1 srun python bigger_exp.py with "$1" acq_name="$2" eps_schedule_id="$3"

# run with sbatch run_slurm.sh CONFIG_NAME OBJ_NAME SEED
