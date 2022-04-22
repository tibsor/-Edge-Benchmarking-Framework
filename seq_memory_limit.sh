#!/bin/bash
bash --version
echo "" 

# from HIGH_VALUE -> LOW_VALUE

memory_limit_values=$(seq 128 -2 126) # range for memory values

lower_limit="${memory_limit_values[-1]}" 
upper_limit="${memory_limit_values[0]}"
echo "Lower memory limit: $lower_limit"
echo "Upper memory limit: $upper_limit"
# sleep 10

lower_limit=126 # FOR DEBUG

for memory_value in $memory_limit_values
do
    mem_limit=$memory_value
    mem_reserve=`echo "$memory_value/2" | bc`
    echo "Starting inference container with $mem_reserve MB memory reserve and $mem_limit MB memory limit"
    #docker run -it --memory="${mem_limit}m" --memory-reservation="${mem_reserve}m" --memory-swap=256m inf_bench:latest python3 main.py  --model_name MLP --data_name SEU --data_dir /inference/Mechanical-datasets --normlizetype mean-std --processing_type O_A --checkpoint_dir /inference/checkpoint
    docker run -it --memory="${mem_limit}m" --memory-reservation="${mem_reserve}m" inf_bench:latest python3 main.py  --model_name MLP --data_name SEU --data_dir /inference/Mechanical-datasets --normlizetype mean-std --processing_type O_A --checkpoint_dir /inference/checkpoint
    var=$?
    if [ $var -eq 0 ]; then 
        echo "Container return value: $var"
        echo "Container ran succesfully! Reducing memory..."
    elif [ $var -eq 137 ]; then # 137 error in Python is OOM
        echo "Container return value: $var"
        echo "Container failed to run!"
        echo "Reason: excessive memory usage"
        PIDS=$(ps -eaf)
        PID=$(echo "$PIDS" | grep "docker_monitor.sh" | awk '{print $2}')
        kill -9 $PID
        break
        elif [ $lower_limit -eq $mem_limit ]; then
        echo "Lower limit reached!"      
        PIDS=$(ps -eaf)
        PID=$(echo "$PIDS" | grep "docker_monitor.sh" | awk '{print $2}')
        kill -9 $PID
        break
    fi
done

