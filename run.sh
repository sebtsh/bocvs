# run with ./run.sh OBJ_NAME ACQ_NAME EPS_SCHEDULE_ID FIRST_CPU LAST_CPU
# run with ./run.sh gpsample ucb-cs 0 0 31
echo "obj_name: $1"
echo "acq_name: $2"
echo "epsilon_schedule: $3"
echo "first cpu: $4"
echo "last cpu: $5"
mkdir out

if [ "$2" == "ucb" ] || [ "$2" == "ts" ]
then
  for seed in {0..4}
  do
    echo "Running $1 exp with $2 seed=$seed"
    CUDA_VISIBLE_DEVICES=-1 taskset -c "$4"-"$5" nohup python exp.py with "$1" acq_name="$2" seed="$seed" > \
    out/"$1_$2_es-0_cost-0_var-0_seed-$seed.txt" &
  done
elif [[ "$2" == "ucb-cs" ]]
then
  for costs_id in {0..2}
  do
    for var_id in {0..2}
    do
      for seed in {0..4}
      do
        echo "Running $1 exp with $2 costs_id=$costs_id var=$var seed=$seed"
        CUDA_VISIBLE_DEVICES=-1 taskset -c "$4"-"$5" nohup python exp.py with "$1" acq_name="$2" \
        eps_schedule_id="$3" costs_id="$costs_id" var_id="$var_id" seed="$seed" > \
        out/"$1_$2_es-$3_cost-${costs_id}_var-${var_id}_seed-$seed.txt" &
      done
    done
  done
fi

#CUDA_VISIBLE_DEVICES=-1 taskset -c 0-31 python exp.py with gpsample acq_name=ucb seed=0