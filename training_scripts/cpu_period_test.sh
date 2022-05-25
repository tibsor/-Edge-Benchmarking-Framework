#!/bin/bash
bash --version
echo ""
source "./local_env/bin/activate" # needed for python
# python will be called to return a decimal point for 
CURRENT_DIR=$(pwd)
echo "Current working directory:$CURRENT_DIR"
echo "Make sure your upper limit is higher than your lower limit!"
echo "Define an upper limit:"
read upper_limit
echo "Define a lower limit:"
read lower_limit
echo "Define a step:"
read step
cpu_quota_values=$(seq $upper_limit -$step $lower_limit)

# Check for old containers running and kill them
docker stop bench_test 1> /dev/null 2> /dev/null
docker rm bench_test 1> /dev/null 2> /dev/null
#redirect stdout and stderr to null for less verbose
#rm $CURRENT_DIR/host_data/*

for cpu_quota in $cpu_quota_values
do
#TODO: add docker monitor process kill
     echo "Starting inference container with CPU quota $cpu_quota"

     docker run --name train_cpu_bench_test -it  -e CPU_QUOTA="$cpu_quota" --memory="1024m" --cpu-period="100000" --cpu-quota="$cpu_quota" \
     -v $CURRENT_DIR/host_data:/inference/volume_data --rm inf_bench:latest \
     python3 train_main_ae.py --model_name Ae1d --data_name SEU --data_dir /inference/Mechanical-datasets --normalizetype mean-std --processing_type O_A
done
# export upper_limit
# export lower_limit
# export step
# python3 plot.py
