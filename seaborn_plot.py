import datetime
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
    train_RAM_log_name = 'RAM_training.log'
    train_CPU_log_name = 'CPU_training.log' 
    global model_dict
    dataset_folders = f'{cwd}/host_data'
    dataset_name = questionary.select("Choose dataset:",choices=os.listdir(dataset_folders)).ask()
    model_dict = {"model": None,"dataset": None, "normalizetype": None,"processing_type": None}
    model_folders = os.path.join(dataset_folders,dataset_name)
    model_name = questionary.select("Choose model folder:", choices=os.listdir(model_folders)).ask()
    working_dir = os.path.join(model_folders,model_name)
    plot_selection = questionary.select("Train or inference:", choices=["train", "inference"]).ask()
    RAM_log_path = os.path.join (working_dir,train_RAM_log_name)
    CPU_log_path = os.path.join(working_dir, train_CPU_log_name)
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


def folder_check(folder_path: str = None):
    isdir = os.path.isdir(folder_path)
    if isdir:
        pass
    else:
        os.mkdir(folder_path)
    


def single_plots(x_axis: list = None, x1_axis: list = None, y_axis: list = None, model_name: str = None, dataset_name: str = None, RAM_flag: bool = False, CPU_flag: bool = False):
    cwd = os.getcwd()
    args_time_list=[]
    init_time_list=[]
    setup_time_list=[]
    eval_time_list=[]
    create_folder_list=[]
    date_now = datetime.date.today()

    plots_folder = os.path.join(cwd,'plots')
    folder_check(plots_folder)
    
    model_folder = os.path.join(plots_folder, model_name)
    folder_check(model_folder)
    
    dataset_folder = os.path.join(model_folder, dataset_name)
    folder_check(dataset_folder)

    date_folder = os.path.join(dataset_folder, str(date_now))
    folder_check(date_folder)

    plots_path=os.path.join(plots_folder, model_name, dataset_name, str(date_now))
    folder_check(plots_path)
    
    total_time=0.0
    for index,i in enumerate(x_axis):
        total_time+=i
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
        setup_df = pd.DataFrame(list(zip(y_axis, setup_time_list)), columns = ['RAM (MB)', 'Setup Time'])
        eval_df = pd.DataFrame(list(zip(y_axis, eval_time_list)), columns = ['RAM (MB)', 'Eval Time'])
        total_df = pd.DataFrame(list(zip(y_axis, x1_axis)), columns = ['RAM (MB)', 'Total Time'])

        ax1 = sns.relplot(x="RAM (MB)", y="Args Time", data=args_df, kind='line', ci=90,markers=True, dashes=False)
        ax2 = sns.relplot(x="RAM (MB)", y="Create Folder Time", kind="line", data=folder_df, ci=90,markers=True, dashes=False)
        ax3 = sns.relplot(x="RAM (MB)", y="Init Time", kind="line", data=init_df, ci=90,markers=True, dashes=False)
        ax4 = sns.relplot(x="RAM (MB)", y="Setup Time", kind="line", data=setup_df, ci=90, markers=True, dashes=False)
        ax5 = sns.relplot(x="RAM (MB)", y="Eval Time", kind="line", data=eval_df, ci=90, markers=True, dashes=False)
        ax6 = sns.relplot(x="RAM (MB)", y="Total Time", kind="line", data=total_df, ci=90, markers=True, dashes=False)
        
        ax1.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Args Time")
        ax2.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Create Folder Time")
        ax3.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Init Time")
        ax4.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Setup Time")
        ax5.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Eval Time")
        ax6.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Total Time")
        plots_path = os.path.join(plots_path, "RAM Results")
        folder_check(plots_path)
        ax1.savefig(f'{plots_path}/{date_now}_RAM_1.png')
        ax2.savefig(f'{plots_path}/{date_now}_RAM_2.png')
        ax3.savefig(f'{plots_path}/{date_now}_RAM_3.png')
        ax4.savefig(f'{plots_path}/{date_now}_RAM_4.png')
        ax5.savefig(f'{plots_path}/{date_now}_RAM_5.png')
        ax6.savefig(f'{plots_path}/{date_now}_RAM_6.png')
    if CPU_flag:
        args_df = pd.DataFrame(list(zip(y_axis, args_time_list)), columns = ['CPU Cores', 'Args Time'])
        folder_df = pd.DataFrame(list(zip(y_axis, create_folder_list)), columns = ['CPU Cores', 'Create Folder Time'])
        init_df = pd.DataFrame(list(zip(y_axis, init_time_list)), columns = ['CPU Cores', 'Init Time'])
        setup_df = pd.DataFrame(list(zip(y_axis, setup_time_list)), columns = ['CPU Cores', 'Setup Time'])
        eval_df = pd.DataFrame(list(zip(y_axis, eval_time_list)), columns = ['CPU Cores', 'Eval Time'])
        total_df = pd.DataFrame(list(zip(y_axis, x1_axis)), columns = ['CPU Cores', 'Total Time'])

        ax1 = sns.relplot(x="CPU Cores", y="Args Time", data=args_df, kind='line', ci=90,markers=True, dashes=False)
        ax2 = sns.relplot(x="CPU Cores", y="Create Folder Time", kind="line", data=folder_df, ci=90,markers=True, dashes=False)
        ax3 = sns.relplot(x="CPU Cores", y="Init Time", kind="line", data=init_df, ci=90,markers=True, dashes=False)
        ax4 = sns.relplot(x="CPU Cores", y="Setup Time", kind="line", data=setup_df, ci=90, markers=True, dashes=False)
        ax5 = sns.relplot(x="CPU Cores", y="Eval Time", kind="line", data=eval_df, ci=90, markers=True, dashes=False)
        ax6 = sns.relplot(x="CPU Cores", y="Total Time", kind="line", data=total_df, ci=90, markers=True, dashes=False)

        ax1.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Args Time")
        ax2.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Create Folder Time")
        ax3.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Init Time")
        ax4.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Setup Time")
        ax5.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Eval Time")
        ax6.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Total Time")

        plots_path = os.path.join(plots_path, "CPU Results")
        folder_check(plots_path)

        ax1.savefig(f'{plots_path}/{date_now}_CPU_1.png')
        ax2.savefig(f'{plots_path}/{date_now}_CPU_2.png')
        ax3.savefig(f'{plots_path}/{date_now}_CPU_3.png')
        ax4.savefig(f'{plots_path}/{date_now}_CPU_4.png')
        ax5.savefig(f'{plots_path}/{date_now}_CPU_5.png')
        ax6.savefig(f'{plots_path}/{date_now}_CPU_6.png')



def main_plot(x_axis: list = None, x1_axis: list = None, y_axis: list = None, model_name: str = None, dataset_name: str = None, RAM_flag: bool = False, CPU_flag: bool = False):
    cwd = os.getcwd()
    args_time_list=[]
    init_time_list=[]
    setup_time_list=[]
    eval_time_list=[]
    create_folder_list=[]
    
    plots_folder = os.path.join(cwd,'plots')
    folder_check(plots_folder)
    
    model_folder = os.path.join(plots_folder, model_name)
    folder_check(model_folder)
    
    dataset_folder = os.path.join(model_folder, dataset_name)
    folder_check(dataset_folder)
    
    plots_path=os.path.join(plots_folder, model_name, dataset_name)
    folder_check(plots_path)
    
    total_time=0.0
    for index,i in enumerate(x_axis):
        total_time+=i
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
    date_now = datetime.date.today()


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
        single_plots(function_time_list, total_time_list, RAM_limit_list, model_name, dataset_name, RAM_flag, CPU_flag)    
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
        single_plots(function_time_list, total_time_list, CPU_limit_list, model_name, dataset_name, RAM_flag, CPU_flag)

    else: print("No CPU file found!")
    
    #plt.show()
    print ("Script ran sucessfully!")
