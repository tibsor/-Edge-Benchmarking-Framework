#!/bin/bash
# NOT WORKING CURRENTLY
bash --version
echo "" 

# from HIGH_VALUE -> LOW_VALUE

#cpu_limit_values=$(seq 1 -0.1 0.5) # range for CPU values
cpu_limit_values=(1 0.9 0.8 0.7 0.6 0.5) #DEBUG
##TBD which is better###
#cpu_period_values=$(seq 100000 -100 100)
#cpu_shares



lower_limit=0.6 # FOR DEBUG

# Check for old containers running and kill them
docker stop bench_test 1> /dev/null 2> /dev/null
docker rm bench_test > /dev/null
#redirect stdout and stderr to null for less verbose

for cpu_value in $cpu_limit_values
do
    cpu_limit=$cpu_value
    echo "Starting inference container with $cpu_limit CPU(s)"
    #docker run -it --memory="${mem_limit}m" --memory-reservation="${mem_reserve}m" --memory-swap=256m inf_bench:latest python3 main.py  --model_name MLP --data_name SEU --data_dir /inference/Mechanical-datasets --normlizetype mean-std --processing_type O_A --checkpoint_dir /inference/checkpoint
    docker run --name bench_test -it --cpus="${cpu_limit}" --rm inf_bench:latest python3 main.py --model_name MLP --data_name SEU --data_dir /inference/Mechanical-datasets --normlizetype mean-std --processing_type O_A --checkpoint_dir /inference/checkpoint
    var=$?
    echo "Run finished! Clean-up..."
    if [ $var -eq 0 ]; then 
        echo "Container return value: $var"
        echo "Container ran succesfully! Reducing CPU limit..."
    elif [ $var -eq 137 ]; then # 137 error in Python is OOM
        echo "Container return value: $var"
        echo "Container failed to run!"
        echo "Reason: excessive CPU usage"
        PIDS=$(ps -eaf)
        PID=$(echo "$PIDS" | grep "docker_monitor.sh" | awk '{print $2}')
        kill -9 $PID
        break
    fi
    # if [ $lower_limit -eq $cpu_limit ] || [ $lower_limit -gt $cpu_limit ]; then
    #     echo "Lower limit reached!"      
    #     PIDS=$(ps -eaf)
    #     PID=$(echo "$PIDS" | grep "docker_monitor.sh" | awk '{print $2}')
    #     kill -9 $PID
    #     break
    # fi
    if (( $(echo "$lower_limit > $cpu_limit" | bc -l) )); then
        echo "Lower limit reached!"      
        PIDS=$(ps -eaf)
        PID=$(echo "$PIDS" | grep "docker_monitor.sh" | awk '{print $2}')
        kill -9 $PID
        break
    fi

done

