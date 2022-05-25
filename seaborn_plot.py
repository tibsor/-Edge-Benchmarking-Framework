import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import questionary
import os
import csv
#sns.set_theme(style="darkgrid")


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


def CPU_function_time_plots(x_axis: list = None, x1_axis: list = None, y_axis: list = None, model_name: str = None, dataset_name: str = None):
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

def main(x_axis: list = None, x1_axis: list = None, y_axis: list = None, model_name: str = None, dataset_name: str = None, RAM_flag: bool = False, CPU_flag: bool = False):
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

    if RAM_flag:
        args_df = pd.DataFrame(list(zip(y_axis, args_time_list)), columns = ['RAM (MB)', 'Args Time'])
        folder_df = pd.DataFrame(list(zip(y_axis, create_folder_list)), columns = ['RAM (MB)', 'Create Folder Time'])
        init_df = pd.DataFrame(list(zip(y_axis, init_time_list)), columns = ['RAM (MB)', 'Init Time'])
        eval_df = pd.DataFrame(list(zip(y_axis, eval_time_list)), columns = ['RAM (MB)', 'Eval Time'])

        ax1 = sns.relplot(x="RAM (MB)", y="Args Time", data=args_df, kind='line', ci=90,markers=True, dashes=False)
        ax2 = sns.relplot(x="RAM (MB)", y="Create Folder Time", kind="line", data=folder_df, ci=90,markers=True, dashes=False)
        ax3 = sns.relplot(x="RAM (MB)", y="Init Time", kind="line", data=init_df, ci=90,markers=True, dashes=False)
        ax4 = sns.relplot(x="RAM (MB)", y="Eval Time", kind="line", data=eval_df, ci=90, markers=True, dashes=False)
        ax1.fig.suptitle("RAM vs Args Time")
        ax2.fig.suptitle("RAM vs Create Folder Time")
        ax3.fig.suptitle("RAM vs Init Time")
        ax4.fig.suptitle("RAM vs Eval Time")

    if CPU_flag:
        args_df = pd.DataFrame(list(zip(y_axis, args_time_list)), columns = ['CPU Cores', 'Args Time'])
        folder_df = pd.DataFrame(list(zip(y_axis, create_folder_list)), columns = ['CPU Cores', 'Create Folder Time'])
        init_df = pd.DataFrame(list(zip(y_axis, init_time_list)), columns = ['CPU Cores', 'Init Time'])
        eval_df = pd.DataFrame(list(zip(y_axis, eval_time_list)), columns = ['CPU Cores', 'Eval Time'])

        ax1 = sns.relplot(x="CPU Cores", y="Args Time", data=args_df, kind='line', ci=90,markers=True, dashes=False)
        ax2 = sns.relplot(x="CPU Cores", y="Create Folder Time", kind="line", data=folder_df, ci=90,markers=True, dashes=False)
        ax3 = sns.relplot(x="CPU Cores", y="Init Time", kind="line", data=init_df, ci=90,markers=True, dashes=False)
        ax4 = sns.relplot(x="CPU Cores", y="Eval Time", kind="line", data=eval_df, ci=90, markers=True, dashes=False)
        ax1.fig.suptitle("CPU Cores vs Args Time")
        ax2.fig.suptitle("CPU Cores vs Create Folder Time")
        ax3.fig.suptitle("CPU Cores vs Init Time")
        ax4.fig.suptitle("CPU Cores vs Eval Time")

if __name__=="__main__":
    cpu_values_list = None
    memory_values_list = None
    cpu_values_list, memory_values_list, model_name, dataset_name = init()
    time_sum=0.0

    RAM_limit_list=[]
    CPU_limit_list=[]
    RAM_flag=False
    CPU_flag=False
    if memory_values_list != None:
        RAM_flag=True
        CPU_flag=False
        function_time_list=[]
        total_time_list=[]
        for runtime_list in memory_values_list:
            RAM_limit_list.append(float(runtime_list[1]))
            for function_time in runtime_list[2:]:
                function_time_list.append(float(function_time))
                time_sum+=float(function_time)
            total_time_list.append(time_sum)
            time_sum=0.0
        main(function_time_list, total_time_list, RAM_limit_list, model_name, dataset_name, RAM_flag, CPU_flag)    
    else: print("No memory file found!")
    if cpu_values_list != None:
        CPU_flag=True
        RAM_flag=False
        function_time_list=[]
        total_time_list = []
        for runtime_list in cpu_values_list:
            CPU_qttc= float(runtime_list[1])/100000.0 # convert cpu quota to cpu cores for easier interpretation
            CPU_limit_list.append(float(CPU_qttc))
            for function_time in runtime_list[2:]:
            #print(time)
                function_time_list.append(float(function_time))
                time_sum+=float(function_time)
            total_time_list.append(time_sum)
            time_sum=0.0
        main(function_time_list, total_time_list, CPU_limit_list, model_name, dataset_name, RAM_flag, CPU_flag)

    else: print("No CPU file found!")
    
    plt.show()
    print ("Script ran sucessfully!")
