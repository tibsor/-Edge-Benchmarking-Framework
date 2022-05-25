from logging import exception
from tkinter.messagebox import NO
import matplotlib.pyplot as plt
import numpy as np # for creating memory range list
import os # for getting bash script variables
import csv 
import questionary


def init():
    cwd = os.getcwd()
    cpu_values_list = None
    memory_values_list = None
    train_cpu_file_name = "train_cpu_quota_runtime_values.csv"
    train_memory_file_name = "train_memory_runtime_values.csv"
    inference_cpu_file_name = "inference_cpu_quota_runtime_values.csv"
    inference_memory_file_name = "inference_memory_runtime_values.csv"
    global model_dict
    model_folders = f'{cwd}/host_data'
    model_name = questionary.select("Choose model folder:", choices=os.listdir(model_folders)).ask()
    model_dict = {"model": None,"dataset": None, "normalizetype": None,"processing_type": None}
    dataset_folders = os.path.join(model_folders,model_name)
    dataset_name = questionary.select("Choose dataset:",choices=os.listdir(dataset_folders)).ask()
    working_dir = os.path.join(dataset_folders,dataset_name)
    plot_selection = questionary.select("Train or inference:", choices=["train", "inference"]).ask()
    if plot_selection == "train":
        cpu_values_file = os.path.join(working_dir, train_cpu_file_name)
        memory_values_file = os.path.join(working_dir, train_memory_file_name)
    else:
        cpu_values_file = os.path.join(working_dir, inference_cpu_file_name)
        memory_values_file = os.path.join(working_dir, inference_memory_file_name)
    if os.path.isfile(cpu_values_file):
        cpu_values_list = []
        with open(cpu_values_file,"r") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            # model_header=next(csvreader, None)  # [model,dataset,normalizetype,processing_type]
            # model_values=next(csvreader, None)
            # model_dict['model']=model_values[0]
            # model_dict['dataset']=model_values[1]
            # model_dict['normalizetype']=model_values[2]
            # model_dict['processing_type']=model_values[3]
            runtime_header = next(csvreader, None)
            for row in csvreader:
                cpu_values_list.append(row)
    if os.path.isfile(memory_values_file):
        memory_values_list = []
        with open(memory_values_file, "r") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            runtime_header = next(csvreader, None)
            for row in csvreader:
                memory_values_list.append(row)

    if cpu_values_list != None and memory_values_list != None:
        return cpu_values_list, memory_values_list, model_name, dataset_name
    elif cpu_values_list != None:
        return cpu_values_list, None, model_name, dataset_name
    elif memory_values_list != None:
        return None,memory_values_list, model_name, dataset_name
    else: return None,None, model_name, dataset_name



def main(x_axis: list = None, y_axis: list = None):
    # if x_axis==None:
    #     raise ValueError("X axis list is none!")
    # if y_axis==None:
    #     raise ValueError("Y axis list is none!")
    main_plot=plt.figure(2)
    plt.plot(x_axis,y_axis, 'ro')
    plt.xlabel('Runtime(seconds)')
    plt.ylabel('RAM (MB)')
    plt.title("Total time elapsed vs RAM usage")

    plt.show()

def RAM_function_time_plots(x_axis: list = None, x1_axis: list = None, y_axis: list = None, model_name: str = None, dataset_name: str = None):
    # if x_axis==None:
    #     raise ValueError("X axis list is none!")
    # if y_axis==None:
    #     raise ValueError("Y axis list is none!")
    args_time_list=[]
    init_time_list=[]
    setup_time_list=[]
    eval_time_list=[]
    create_folder_list=[]
    for index,i in enumerate(x_axis):
        if index%5==0:
            args_time_list.append(i)
        elif index%5==1:
            create_folder_list.append(i)
        elif index%5==2:
            init_time_list.append(i)
        elif index%5==3:
            setup_time_list.append(i)
        elif index%5==4:
            eval_time_list.append(i)

    fig, axs = plt.subplots(3, 2)
    fig.suptitle('Function time vs RAM usage')

    axs[0, 0].plot(args_time_list, y_axis, marker='v', linestyle='')
    axs[0, 0].set_title('parse_args()')
    axs[0, 1].plot(create_folder_list, y_axis, 'tab:orange', marker='.', linestyle='')
    axs[0, 1].set_title('create_folder()')
    axs[1, 0].plot(init_time_list, y_axis, 'tab:green',marker='o', linestyle='')
    axs[1, 0].set_title('inference.init()')
    axs[1, 1].plot(setup_time_list, y_axis, 'tab:red', marker='^', linestyle='')
    axs[1, 1].set_title('inference.setup()')
    axs[2, 0].plot(eval_time_list, y_axis, marker='o', linestyle='')
    axs[2, 0].set_title('inference.eval()')
    axs[2, 1].plot(x1_axis, y_axis, marker='o', linestyle='')
    axs[2, 1].set_title('Total Time')
    fig.tight_layout()
    for ax in axs.flat:
        ax.set(xlabel='Time (s)', ylabel='RAM (MB)')
    #plt.savefig("fct_rt.png")
    plt.show(block=False)

def CPU_function_time_plots(x_axis: list = None, x1_axis: list = None, y_axis: list = None, model_name: str = None, dataset_name: str = None):
    # if x_axis==None:
    #     raise ValueError("X axis list is none!")
    # if y_axis==None:
    #     raise ValueError("Y axis list is none!")
    args_time_list=[]
    init_time_list=[]
    setup_time_list=[]
    eval_time_list=[]
    create_folder_list=[]
    for index,i in enumerate(x_axis):
        if index%5==0:
            args_time_list.append(i)
        elif index%5==1:
            create_folder_list.append(i)
        elif index%5==2:
            init_time_list.append(i)
        elif index%5==3:
            setup_time_list.append(i)
        elif index%5==4:
            eval_time_list.append(i)

    fig, axs = plt.subplots(3, 2)
    fig.suptitle(f'Model:{model_name}\nDataset:{dataset_name}\nFunction time vs CPU Quota')

    axs[0, 0].plot(args_time_list, y_axis, marker='v', linestyle='')
    axs[0, 0].set_title('parse_args()')
    axs[0, 1].plot(create_folder_list, y_axis, 'tab:orange', marker='.', linestyle='')
    axs[0, 1].set_title('create_folder()')
    axs[1, 0].plot(init_time_list, y_axis, 'tab:green',marker='o', linestyle='')
    axs[1, 0].set_title('inference.init()')
    axs[1, 1].plot(setup_time_list, y_axis, 'tab:red', marker='^', linestyle='')
    axs[1, 1].set_title('inference.setup()')
    axs[2, 0].plot(eval_time_list, y_axis, marker='o', linestyle='')
    axs[2, 0].set_title('inference.eval()')
    axs[2, 1].plot(x1_axis, y_axis, marker='o', linestyle='')
    axs[2, 1].set_title('Total Time')
    fig.tight_layout()
    for ax in axs.flat:
        ax.set(xlabel='Time (s)', ylabel='CPU Quota')
    #plt.savefig("fct_rt.png")
    plt.show(block=False)


if __name__=="__main__":
    cpu_values_list = None
    memory_values_list = None
    cpu_values_list, memory_values_list, model_name, dataset_name = init()
    time_sum=0.0

    RAM_limit_list=[]
    CPU_limit_list=[]
    if memory_values_list != None:
        function_time_list=[]
        total_time_list=[]
        for runtime_list in memory_values_list:
            RAM_limit_list.append(float(runtime_list[1]))
            for function_time in runtime_list[2:]:
            #print(time)
                function_time_list.append(float(function_time))
                time_sum+=float(function_time)
            total_time_list.append(time_sum)
            time_sum=0.0
        RAM_function_time_plots(function_time_list, total_time_list, RAM_limit_list, model_name, dataset_name)
    else: print("No memory file found!")
    if cpu_values_list != None:
        function_time_list=[]
        total_time_list = []
        for runtime_list in cpu_values_list:
            CPU_limit_list.append(float(runtime_list[1]))
            for function_time in runtime_list[2:]:
            #print(time)
                function_time_list.append(float(function_time))
                time_sum+=float(function_time)
            total_time_list.append(time_sum)
            time_sum=0.0
        CPU_function_time_plots(function_time_list, total_time_list, CPU_limit_list, model_name, dataset_name)
    else: print("No CPU file found!")
    
    plt.show()
    #main(, RAM_limit_list)