# Edge Benchmarking Framework

## Getting started

This repository contains the source code and instructions for building and deploying an evaluation/training benchmark framework, currently being able to run the following models and datasets:
1. Models
    * MLP
    * ResNet
    * CNN
    * LeNet
    * AlexNet
    * Sae1d
    * Ae1d
2. Datasets
    * SEU
    * MFPT
    * CWRU
    * PU (bad results, future testing required)

## Requirements
Linux-based OS

Docker 
- [installation page](
https://docs.docker.com/desktop/linux/install/)



## First time use
Please follow these instructions carefully in order to ensure propper framework setup.

1. Start by configuring the Dockerfile to your needs.

Inside the file you add the comand to copy the dataset you require to evaluate in the `DATASET SECTION`.

2. Next, build the docker image
`docker build -t bench_fw:latest .`

Note that the name of the image (`bench_fw`) is important, as it will be required by the bash script later on.

3. You can start the framework by calling one of the following scripts: 
- `train_menu_ram.sh`
- `train_menu_cpu.sh`
- `inference_menu_ram.sh`
- `inference_menu_cpu.sh`

In order to run the inference script, first you must run a full training process in order for a model to be trained and saved.

IMPORTANT: The python script running inside the container requires a `MEM_LIMIT` or `CPU_QUOTA` environment value in order to properly write on the shared volume data.

4. Create a local python env with the `plot_env.txt` file and then [activate it](https://docs.python.org/3/tutorial/venv.html). Afterwards, run the `seaborn_plot.py` to see get total runtime plots as well as time-per-epoch plots.
## Description
At the moment, given the image built from the present Dockerfile, the container will run either an inference observation or a complete training process. 

The shell scripts recursively run the container, checks the return value (0 - succesful run, error otherwise), then gradually decreases RAM/CPU size.

There is a `docker_monitor.sh` that runs the `docker stats` command and saves the output, alongside the date and time, in the `benchmark_monitor.txt` file, if you wish to run that alongside your containers.

## Roadmap
- [x] check function peak memory usage - partly done - plots next
- [x] add CPU benchmarking 
- [x] create shared volume to be used by both inference & train benchmark
- [x] add other model types and compare memory usage results
- [x] add other datasets
- [ ] analyze `--memory-swap` option (benefits/drawbacks)


## License
GPL3 License
## Project status
Active


