echo "job_id: $1"

ulimit -u 100000
ulimit -l unlimited
ulimit -d unlimited
ulimit -m unlimited
ulimit -v unlimited

CUDA_VISIBLE_DEVICES=-1 python job_runner.py "$1"
