#!/bin/bash
bash --version
echo ""
source "./local_env/bin/activate" # needed for python
# python will be called to return a decimal point for 

CURRENT_DIR=$(pwd)
echo $CURRENT_DIR
upper_limit=1 # CPU cores
lower_limit=5 # given in CPU cores/10 if lower_limit > upper_limit
limit_adjustment=10 # used IF lower_limit > upper_limit
scale=100 
# TODO: implement scale check
step=1 # define as integer because of bash float integration, it's actually 0.1
# Check for old containers running and kill them
docker stop bench_test 1> /dev/null 2> /dev/null
docker rm bench_test 1> /dev/null 2> /dev/null
#redirect stdout and stderr to null for less verbose

rm $CURRENT_DIR/host_data/*

for ((i=$upper_limit*$scale; i >=$lower_limit*100/$limit_adjustment; i -= $step))
do
     #printf %.10f\\n "$((1000000000 *   20/7  ))e-9"
     echo "Starting inference container with $(python -c "print(${i}/$scale)") total CPU count"

     #cpu_value=$( bc <<< "${i}/100")
     docker run --name bench_test -it --cpus="$(python -c "print(${i}/$scale)")" -v $CURRENT_DIR/host_data:/inference/volume_data --rm inf_bench:latest python3 main.py --model_name MLP --data_name SEU --data_dir /inference/Mechanical-datasets --normlizetype mean-std --processing_type O_A --checkpoint_dir /inference/checkpoint
done
# export upper_limit
# export lower_limit
# export step
# python3 plot.py
