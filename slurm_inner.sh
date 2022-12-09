echo "obj_name: $1"
echo "acq_name: $2"
echo "eps_schedule_id: $3"

ulimit -u 100000
ulimit -l unlimited
ulimit -d unlimited
ulimit -m unlimited
ulimit -v unlimited

CUDA_VISIBLE_DEVICES=-1 python bigger_exp.py "$1" "$2" "$3"
