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
        echo "Starting inference container with CPU quota $cpu_quota"

        docker run --rm -it -e CPU_QUOTA="$cpu_quota" --memory="2048m" --cpu-period="100000" --cpu-quota="$cpu_quota" -v $CURRENT_DIR/host_data:/benchmark/volume_data bench_fw:latest python3 train_main.py --model_name $1 --data_name $2 --normalizetype mean-std --processing_type O_A --max_epoch 10 --middle_epoch 10
    done
    done
}

PS3="Select the dataset/model combination: "
echo $benchmark
select dataset in SEU MFPT; do
  case $dataset in
    SEU)
    select model in MLP Alexnet1d Resnet1d CNN_1d LeNet1d Sae1d Ae1d; do
        CPU_benchmark $model $dataset
        echo "C'est fini"
        #break
    break
    done
    ;;
    MFPT)
    select model in  MLP Alexnet1d Resnet1d CNN_1d LeNet1d; do
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
