#!/bin/bash

function CPU_benchmark {
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
    for i in {0..4..1}
        do 
    #TODO: add docker monitor process kill
        echo "Dataset/Model:$2/$1"
        echo "Starting benchmark container with CPU quota $cpu_quota"

        docker run --rm -it -e CPU_QUOTA="$cpu_quota" --memory="2048m" --cpu-period="100000" --cpu-quota="$cpu_quota" -v $CURRENT_DIR/host_data:/benchmark/volume_data bench_fw:latest python3 train_main.py --model_name $1 --data_name $2 --normalizetype mean-std --processing_type O_A --max_epoch 10 --middle_epoch 10
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
    done
    done
}



PS3="Select the dataset/model combination: "
echo $benchmark
select dataset in SEU MFPT CWRU; do
  case $dataset in
    SEU)
    select model in Alexnet1d Resnet1d CNN_1d LeNet1d BiLSTM1d Sae1d Ae1d MLP; do
        CPU_benchmark $model $dataset
        echo "C'est fini"
        #break
    break
    done
    ;;
    MFPT)
    select model in Alexnet1d Resnet1d CNN_1d LeNet1d BiLSTM1d Sae1d Ae1d MLP; do
        CPU_benchmark $model $dataset
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
        CPU_benchmark $model $dataset
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
