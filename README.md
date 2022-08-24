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
https://stackoverflow.com/questions/48957195/how-to-fix-docker-got-permission-denied-issue
Make sure you have installed python pip with `sudo apt install python3-pip`
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

## Alternative start
Create a local python env with the `plot_env.txt` file and then [activate it](https://docs.python.org/3/tutorial/venv.html). Then run `python main.py` and follow the on screen instructions.

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


## Notes
The following guidelines apply to systems which can run some sort of GNU/Linux OS and support the Docker software. One should start by downloading the repository available on GitLab. Within this repository's folder the only thing missing in order to start the application is the dataset, however the SEU dataset is provided (as a git submodule) for a getting started example. The 'src' folder is the source code that will be running as a containerized application with the Docker daemon. That is where we can find the scripts for running either training or inference, the currently available models and datasets, alongside a few 'utils' scripts and an environment file. Future models should be added within the 'src/models' folder, by saving the model with its corresponding name so it can be parsed as an argument. The same logic applies for datasets, but under what is currently 'src/CNN\_Datasets/*data split type*'. There are 3 types of data splits available: organized augmented, randomized augmented and randomized non augmented.
When running, the working directory within the container is called "benchmark" and it is inside this folder, at location "benchmark/volume\_data", that our data will be saved and shared with the host machine. 
Outside the 'src' folder we have our Docker file, which acts as the means to build our containerized application. Bash scripts are provided for running either training or inference alongside a script for plotting the values that one will obtain by running the benchmark process.
The first step for getting started is adding your dataset inside the "Dockerfile" within the main folder. Choose your favourite text editor, open the file and based on the examples already present there, add the line to copy your dataset to the container. Save the file, and make sure when building the framework image that you respect the naming convention mentioned inside the 'README.md' file, as the image name is crucial. The Docker file takes care of all the other dependencies needed (packages, creating and activating the environment). After the build is done, it is recommended to start by running one of the two training scripts provided, which will save the best model available at the end of its run, making it available for inference. The default training will run for 10 epochs, however you can easily modify this and other runtime parameters (such as normalization type, batch size, optimizer) by editing the 'docker run' command arguments inside the training scripts. When you start the application, the upper limit, lower limit and step size must be defined, by console input, according to the prompt (memory in megabytes for the RAM script and cpu quota for the CPU script). Starting from the top limit, these values will be passed to the container as an environment variable, which we check within the isolated application and then write the runtime results according to what the process is (train/inference, RAM/CPU benchmarking). As a general remark, a docker container cannot run with less than 6MB of RAM or a cpu quota of less than 1000