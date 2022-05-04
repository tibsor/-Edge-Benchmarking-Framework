#!/bin/bash
bash --version
echo ""
source "./local_env/bin/activate" # needed for python
# python will be called to return a decimal point for 
CURRENT_DIR=$(pwd)
echo $CURRENT_DIR

# Check for old containers running and kill them
docker stop bench_test 1> /dev/null 2> /dev/null
docker rm bench_test 1> /dev/null 2> /dev/null
#redirect stdout and stderr to null for less verbose

rm $CURRENT_DIR/host_data/*

for cpu_quota in $(seq 100000 -1000 1000)
do
     echo "Starting inference container with CPU quota $cpu_quota"

     docker run --name bench_test -it --cpu-period="100000" --cpu-quota="$cpu_quota" -v $CURRENT_DIR/host_data:/inference/volume_data --rm inf_bench:latest python3 main.py --model_name MLP --data_name SEU --data_dir /inference/Mechanical-datasets --normlizetype mean-std --processing_type O_A --checkpoint_dir /inference/checkpoint
done
# export upper_limit
# export lower_limit
# export step
# python3 plot.py
