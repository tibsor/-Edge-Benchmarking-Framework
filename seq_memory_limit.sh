#!/bin/bash
bash --version
echo "" 
CURRENT_DIR=$(pwd)
echo $CURRENT_DIR
# from HIGH_VALUE -> LOW_VALUE
rm $CURRENT_DIR/host_data/*
memory_limit_values=$(seq 256 -2 64) # range for memory values
upper_limit=256
lower_limit=64 # FOR DEBUG
step=2
# Check for old containers running and kill them
docker stop bench_test 1> /dev/null 2> /dev/null
docker rm bench_test 1> /dev/null 2> /dev/null
#redirect stdout and stderr to null for less verbose

for memory_value in $memory_limit_values
do
    mem_limit=$memory_value
    mem_reserve=`echo "$memory_value/2" | bc`
    echo "Starting inference container with $mem_reserve MB memory reserve and $mem_limit MB memory limit"
    #docker run -it --memory="${mem_limit}m" --memory-reservation="${mem_reserve}m" --memory-swap=256m inf_bench:latest python3 main.py  --model_name MLP --data_name SEU --data_dir /inference/Mechanical-datasets --normlizetype mean-std --processing_type O_A --checkpoint_dir /inference/checkpoint
    docker run --name bench_test -it --cpus="1.0" --memory="${mem_limit}m" --memory-reservation="${mem_reserve}m" --memory-swap=256m -v $CURRENT_DIR/host_data:/inference/volume_data --rm inf_bench:latest python3 main.py --model_name MLP --data_name SEU --data_dir /inference/Mechanical-datasets --normlizetype mean-std --processing_type O_A --checkpoint_dir /inference/checkpoint
    run_output=$?
    echo "Run finished! Clean-up..."
    if [ $run_output -eq 0 ]; then 
        echo "Container return value: $run_output"
        echo "Container ran succesfully! Reducing memory..."
    elif [ $run_output -eq 137 ]; then # 137 error in Python is OOM
        echo "Container return value: $run_output"
        echo "Container failed to run!"
        echo "Reason: excessive memory usage"
        PIDS=$(ps -eaf)
        PID=$(echo "$PIDS" | grep "docker_monitor.sh" | awk '{print $2}')
        kill -9 $PID
        break
    fi
    if [ $lower_limit -eq $mem_limit ] || [ $lower_limit -gt $mem_limit ]; then
        echo "Lower limit reached!"      
        PIDS=$(ps -eaf)
        PID=$(echo "$PIDS" | grep "docker_monitor.sh" | awk '{print $2}')
        kill -9 $PID
        break
    fi
done
#variables to be used for plotting script
source "./local_env/bin/activate"
export upper_limit
export lower_limit
export step
python3 plot.py
