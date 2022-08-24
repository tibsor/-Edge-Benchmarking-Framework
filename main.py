####WRITING SCRIPT FILES####
import subprocess # used for calling bash commands 
import sys
import time # for exiting in case of error
import questionary # used for TUI
import os
import json # keeping track of built docker images
from datetime import datetime
#from tqdm import tqdm
cwd = os.getcwd()
datasets_path = os.path.join(cwd, 'datasets')
models_path = os.path.join(cwd,'src/models')
image_name = 'bench_fw'
tag = 'latest'
docker_file_name = 'Dockerfile_dev'

dockerfile_strings = ['FROM python:3.7.13-slim-buster AS base',
'ENV VIRTUAL_ENV=/opt/benchmark_env', 
'RUN python3 -m venv /opt/benchmark_env', 
'RUN /opt/benchmark_env/bin/python3 -m pip install --upgrade pip',
'ENV PATH="$VIRTUAL_ENV/bin:$PATH"',
'WORKDIR /benchmark/',
'RUN apt update',
'RUN apt install -y libglib2.0-0 libsm6 libxrender1 libxext6',
'COPY ./src/container_env.txt /benchmark/requirements.txt',
'RUN pip install -r requirements.txt',
'COPY  ./src/ /benchmark/']

cpu_top = os.cpu_count()
cpu_bottom = 0.01
mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448
ram_top = int(mem_bytes/(1024.**2))  # MAX RAM memory in MB
ram_bottom = 6 # minimum in MB required to run a docker container
cpu_step = 0.1 # in CPU cores
ram_step = 100 # in MB
benchmark_options = ['cpu', 'ram']
dataset_folder_name = {"SEU":"Mechanical-datasets", "MFPT":"MFPT_Fault_Data_Sets", "CWRU": "CWRU", "PU":"Paderborn", "XJTU":"XJTU-SY_Bearing_Datasets"}

def folder_create(folder_path: str = None):
    isdir = os.path.isdir(folder_path)
    if isdir:
        pass
    else:
        os.mkdir(folder_path)
        

# function to return only files from directory
def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

# function to return all models from source code/model folders
def get_models(path):
    models_list = []
    for file in files(path):
        models_list.append(os.path.splitext(file)[0]) # split text to remove .py extension
    return models_list

def folder_to_dataset(folders_list):
    tmp_list = [] 
    for folder in folders_list:
        for key,value in dataset_folder_name.items():
            if value == folder:
                tmp_list.append(key)
    return tmp_list
#function to list directories (datasets), select which to use and return them
def get_dataset_folders(path: str = None):
    _tmp_list = os.listdir(path)
    folders_list = questionary.checkbox("Selects datasets to use:", choices=_tmp_list).ask()
    return folders_list


def build_image():
    image_name = 'bench_fw'
    tag = 'latest'
    datasets = get_dataset_folders(datasets_path)    
    
    for dataset in datasets:
        dockerfile_strings.append(f'COPY ./datasets/{dataset} /benchmark/{dataset}')
    write_dockerfile(dockerfile_strings)
    if questionary.confirm(f'Change image name? Currently: "{image_name}"').ask():
        image_name = questionary.text("Docker Image name: ").ask()
    if questionary.confirm('Custom image tag?').ask():
        tag = questionary.text("Image tag: ").ask()
    build_output = os.system(f"docker build -t {image_name}:{tag} -f {dockerfile_path} .")
    if build_output == 0 :
        print ("Image succesfully built!\nSaving changes locally...\n")
        # _tmp = {
        #         "timedate": datetime.now(),
        #         "image name": f"{image_name}",
        #         "tag": f"{tag}",
        #         "datasets": datasets
        #         }
        # json_obj = json.dumps(_tmp, indent = 4)
        # del _tmp
        # with open(f'{cwd}/image_index.json', 'a+') as file:
        #     file.write(json_obj)
    else:
        print ("Failed to build image!")
        sys.exit(1)
    return datasets, image_name

def write_dockerfile(strings_to_write):
    global dockerfile_path
    
    dockerfile_path = os.path.join(cwd, docker_file_name)
    with open(dockerfile_path, 'w') as file:
        for line in dockerfile_strings:
            file.write(f'{line}\n')

def search_dataset(path):
    # if questionary.confirm(f'Search "{path}" for datasets?').ask():
    datasets = get_dataset_folders(path)
    # else:
    # os.system("stty sane")
    # tmp_path = input("Input absolute path:")
    # datasets = get_dataset_folders(tmp_path)
    return datasets


def get_iterations():
    if questionary.confirm("Iterate over the same value multiple times?").ask():
        os.system("stty sane")
        while True:
            try:
                train_iterations = int(input("No. of train iterations:"))
                inf_iterations = int(input("No. of inference iterations:"))
            except ValueError:
                print("Sorry, I didn't understand that.")
                continue
            if train_iterations < 0 or inf_iterations < 0:
                print("Sorry, your response must not be negative.")
                continue
            else:
                break
    else:
        train_iterations = 1
        inf_iterations = 1
    return train_iterations, inf_iterations

def get_cpu_limits():
    os.system("stty sane")
    while True:
        try:
            value_top = float(input(f"CPU cores top limit (max={os.cpu_count()}): "))
            value_bottom = float(input("CPU cores bottom limit (min=0.01): "))
            step = float(input("Step size (e.g 0.2): "))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue
        if value_top < 0 or value_bottom < 0 or step < 0:
            print("Sorry, your response must not be negative.")
            continue
        elif value_bottom > value_top:
            print("Sorry, top limit must be higher")
        else:
            break
    value_top = int(value_top * 100000)
    value_bottom = int(value_bottom * 100000)
    step= int(step * 100000)
    return value_top, value_bottom, step

def get_ram_limits():
    os.system("stty sane")
    while True:
        try:
            value_top = int(input(f"RAM(MB) top limit (max={int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')/(1024.**2))}): "))
            value_bottom = int(input("RAM(MB) bottom limit (min=6): "))
            step = int(input("Step size (e.g 100): "))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue
        if value_top < 0 or value_bottom < 0 or step < 0:
            print("Sorry, your response must not be negative.")
            continue
        elif value_bottom > value_top:
            print("Sorry, top limit must be higher")
        else:
            break
    return value_top, value_bottom, step

def write_cpu_scripts(image_name, dataset_list, models, topl_list, bottoml_list, st_list, train_iterations, inf_iterations):
    dataset_lst=(' '.join(dataset_list))
    model_lst=(' '.join(models))
    for process in ['train', 'inference']:
        cpu_script_strings = []
        if process == 'inference':
            iterations = inf_iterations
            topl = topl_list[-1]
            bottoml = bottoml_list[-1]
            st = st_list[-1]
        else:
            topl = topl_list[0]
            bottoml = bottoml_list[0]
            st = st_list[0]
            iterations = train_iterations
        cpu_script_strings = [
            f'cpu_quota_values=$(seq {topl} -{st} {bottoml})',
            f'for dataset in {dataset_lst}',
            'do',
            f'for model in {model_lst}',
            'do',
            'for cpu_quota in $cpu_quota_values',
            'do',
            f'for i in {{0..{iterations}..1}}',
            'do',
            f'echo "Starting {process} container with CPU quota $cpu_quota"',
            'docker run --rm -e CPU_QUOTA="$cpu_quota" --memory="1024m" '
            f'--cpu-period="100000" --cpu-quota="$cpu_quota" -v {cwd}/host_data:/benchmark/volume_data '
            f'{image_name}:latest python3 {process}_main.py ' 
            '--model_name $model --data_name $dataset --normalizetype mean-std --processing_type O_A --max_epoch 1 --middle_epoch 1',
            'done',
            'done',
            'done',
            'done'
            ]
        with open(f'{cwd}/scripts/{process}_cpu.sh', 'w') as file:
            for line in cpu_script_strings:
                file.write(f'{line}\n')

def write_ram_scripts(image_name, dataset_list, models, topl_list, bottoml_list, st_list, train_iterations, inf_iterations):
    dataset_lst=(' '.join(dataset_list))
    model_lst=(' '.join(models))
    for process in ['train', 'inference']:
        ram_script_strings = []
        if process == 'inference':
            iterations = inf_iterations
            topl = topl_list[-1]
            bottoml = bottoml_list[-1]
            st = st_list[-1]
        else:
            topl = topl_list[0]
            bottoml = bottoml_list[0]
            st = st_list[0]
            iterations = train_iterations
        ram_script_strings = [
        f'memory_limit_values=$(seq {topl} -{st} {bottoml})',
        f'for dataset in {dataset_lst}',
        'do',
        f'for model in {model_lst}',
        'do',
        'for mem_limit in $memory_limit_values',
        'do',
        f'for i in {{0..{iterations}..1}}',
        'do',
        f'echo "Starting {process} container with memory limit $mem_limit"',
        'docker run --rm -e MEM_LIMIT="$mem_limit" --memory="${mem_limit}m" '
        f'--cpus="1.0" -v {cwd}/host_data:/benchmark/volume_data {image_name}:latest ' 
        f'python3 {process}_main.py --model_name $model --data_name $dataset ' 
        '--normalizetype mean-std --processing_type O_A --max_epoch 1 --middle_epoch 1',
        'run_output=$?',
        'echo "Run finished! Clean-up..."',
        'if [ $run_output -eq 0 ]; then ',
        'echo "Container return value: $run_output"',
        'echo "Container ran succesfully!"',
        'sleep 2',
        'elif [ $run_output -eq 137 ]; then',
        'echo "Container return value: $run_output"',
        'echo "Container failed to run!"',
        'echo "Reason: excessive memory usage"',
        'fi',
        'done',
        'done',
        'done',
        'done'
        ]
        with open(f'{cwd}/scripts/{process}_ram.sh', 'w') as file:
            for line in ram_script_strings:
                file.write(f'{line}\n')
    

def start_benchmark(benchmark_options):
    for process in ['train', 'inference']:
        for param in benchmark_options:
            makeExecutable = f'chmod +x {cwd}/scripts/{process}_{param}.sh'
            call_subprocess(makeExecutable)
            bashCommand = f'{cwd}/scripts/{process}_{param}.sh'
            benchmark = call_subprocess(bashCommand)

def create_scripts(benchmark_options, image_name, dataset_folders, models):
    scripts_path = os.path.join(cwd,'scripts')
    folder_create(scripts_path)
    for f in os.listdir(scripts_path):
        os.remove(os.path.join(scripts_path, f))
    train_iterations, inf_iterations = get_iterations()
    different_limits = questionary.confirm("Different limits for each process?\ni.e custom top, bottom and step for train AND inference").ask()
    for option in benchmark_options:
        if option == 'cpu':
            if different_limits:    
                print('Train limits first: ')
                cpu_top_train, cpu_bottom_train, cpu_step_train = get_cpu_limits()
                print('Inference limits: ')
                cpu_top_inf, cpu_bottom_inf, cpu_step_inf = get_cpu_limits()
                cpu_top = [cpu_top_train, cpu_top_inf]
                cpu_bottom = [cpu_bottom_train, cpu_bottom_inf]
                cpu_step = [cpu_step_train, cpu_step_inf]
            else:
                print("Same limits will be used for both training and inference")
                cpu_top_train, cpu_bottom_train, cpu_step_train = get_cpu_limits()
                cpu_top_inf = cpu_top_train
                cpu_bottom_inf = cpu_bottom_train
                cpu_step_inf = cpu_step_train
                cpu_top = [cpu_top_train, cpu_top_inf]
                cpu_bottom = [cpu_bottom_train, cpu_bottom_inf]
                cpu_step = [cpu_step_train, cpu_step_inf]

            write_cpu_scripts(image_name, folder_to_dataset(dataset_folders), models, cpu_top, cpu_bottom, cpu_step, train_iterations, inf_iterations)
        if option == 'ram':
            if different_limits:
                print('Train limits first: ')
                ram_top_train, ram_bottom_train, ram_step_train = get_ram_limits()
                print('Inference limits: ')
                ram_top_inf, ram_bottom_inf, ram_step_inf = get_ram_limits()
                ram_top = [ram_top_train, ram_top_inf]
                ram_bottom = [ram_bottom_train, ram_bottom_inf]
                ram_step = [ram_step_train, ram_step_inf]
            else:
                print("Same limits will be used for both training and inference")
                ram_top_train, ram_bottom_train, ram_step_train = get_ram_limits()
                ram_top_inf = ram_top_train
                ram_bottom_inf = ram_bottom_train
                ram_step_inf = ram_step_train
                ram_top = [ram_top_train, ram_top_inf]
                ram_bottom = [ram_bottom_train, ram_bottom_inf]
                ram_step = [ram_step_train, ram_step_inf]
            write_ram_scripts(image_name, folder_to_dataset(dataset_folders), models, ram_top, ram_bottom, ram_step, train_iterations, inf_iterations)

def call_subprocess(command):
    result = subprocess.run(['bash', command])
    return result
def read_json():
    with open(f'{cwd}/image_index.json','r') as file:
        json_data = json.load(file)


if __name__ == "__main__":
    print ("Welcome!")
    print ("This script should be used for writing/building your Dockerfile and setting up the models to benchmark")
    if questionary.confirm("Build new image?").ask():
        dataset_folders, image_name = build_image()
    else:
        # read_json()
        dataset_folders = get_dataset_folders(datasets_path)
    models_list = get_models(models_path)
    models = questionary.checkbox("Select models to benchmark:", choices=models_list).ask()
    benchmark_options = questionary.checkbox ("CPU/RAM benchmarking?",choices=["cpu", "ram"]).ask()
    create_scripts(benchmark_options, image_name, dataset_folders, models)
    start_benchmark(benchmark_options)
    from . import seaborn_plot
