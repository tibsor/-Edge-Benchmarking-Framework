#!/bin/bash
function RAM_benchmark {
    CURRENT_DIR=$(pwd)
    echo "Working directory is:$CURRENT_DIR"
    # from HIGH_VALUE -> LOW_VALUE
    #rm $CURRENT_DIR/host_data/*
    echo "Please make sure the upper limit is higher than the lower limit!"
    echo "Upper limit number (MB):"
    read upper_limit
    echo "Lower limit number (MB):"
    read lower_limit
    echo "Step:"
    read step
    memory_limit_values=$(seq $upper_limit -$step $lower_limit) # range for memory values

    for memory_value in $memory_limit_values
    do
        mem_limit=$memory_value
        # for i in {0..2..1}
            # do 
            echo "Dataset/Model:$2/$1"
            echo "Starting benchmark container with $mem_limit MB memory limit"
            docker run --rm -it -e MEM_LIMIT="$mem_limit" --memory="${mem_limit}m" --cpus="1.0" -v $CURRENT_DIR/host_data:/benchmark/volume_data bench_fw:latest python3 train_main.py --model_name $1 --data_name $2 --normalizetype mean-std --processing_type O_A --max_epoch 10 --middle_epoch 10
            run_output=$?
            echo "Run finished! Clean-up..."
            if [ $run_output -eq 0 ]; then 
                echo "Container return value: $run_output"
                echo "Container ran succesfully!"
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
        # done
    done
}

PS3="Select the dataset/model combination: "
echo $benchmark
select dataset in SEU MFPT CWRU; do
  case $dataset in
    SEU)
    select model in Alexnet1d Resnet1d CNN_1d LeNet1d BiLSTM1d Sae1d Ae1d MLP; do
        RAM_benchmark $model $dataset
        echo "C'est fini"
        #break
    break
    done
    ;;
    MFPT)
    select model in Alexnet1d Resnet1d CNN_1d LeNet1d BiLSTM1d Sae1d Ae1d MLP; do
        RAM_benchmark $model $dataset
        echo "C'est fini"
        #break
    done
    break
    ;;
    quit)
    break
      ;;
    CWRU)
    select model in Alexnet1d Resnet1d CNN_1d LeNet1d BiLSTM1d; do
        RAM_benchmark $model $dataset
        echo "C'est fini"
        #break
    done
    break
    ;;
    quit)
    break
      ;;
    
    *) 
    echo "Invalid option $REPLY"
      ;;
  esac
done
