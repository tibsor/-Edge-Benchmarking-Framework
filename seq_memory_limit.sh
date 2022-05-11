#!/bin/bash
bash --version
echo "" 
CURRENT_DIR=$(pwd)
echo "Working directory is:$CURRENT_DIR"

# from HIGH_VALUE -> LOW_VALUE
echo "Looking for files from older runs..."
#rm $CURRENT_DIR/host_data/*
echo "Please make sure the upper limit is higher than the lower limit!"
echo "Upper limit number (MBs):"
read upper_limit
echo "Lower limit number (MBs):"
read lower_limit
echo "Step:"
read step
memory_limit_values=$(seq $upper_limit -$step $lower_limit) # range for memory values
# upper_limit=256
# lower_limit=254 # FOR DEBUG
# step=2
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
    docker run --name bench_test -it -e MEM_LIMIT="$mem_limit" \
    --memory="${mem_limit}m" \
    --cpus="1.0" -v $CURRENT_DIR/host_data:/inference/volume_data \
    --rm inf_bench:latest python3 main.py\
    --model_name MLP --data_name SEU --data_dir /inference/Mechanical-datasets --normalizetype mean-std --processing_type O_A --checkpoint_dir /inference/checkpoint
    run_output=$?
    echo "Run finished! Clean-up..."
    if [ $run_output -eq 0 ]; then 
        echo "Container return value: $run_output"
        echo "Container ran succesfully! Reducing memory..."
        sleep 2
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

source "./local_env/bin/activate"
python3 plot.py
