#!/bin/bash


memory_limit_values=$(seq 124 -10 12)
# memory_reserve_values=($(seq 1024 -10 2))


for index in $memory_limit_values
do
mem_limit=$index
mem_reserve=`echo "$index/2" | bc`

echo "Starting inference container with $mem_reserve memory reserve and $mem_limit memory limit"
docker run -it --memory="${mem_limit}m" --memory-reservation="${mem_reserve}m" inf_bench:latest python3 main.py  --model_name MLP --data_name SEU --data_dir /inference/Mechanical-datasets --normlizetype mean-std --processing_type O_A --checkpoint_dir /inference/checkpoint
echo "Container ran succesfully! Reducing memory...\n"

# echo "New memory limit:$mem_limit"
# echo "New memory reserve:$mem_reserve"
done
