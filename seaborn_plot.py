import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import questionary
import os
import csv
import re
import time
import bisect


def init():
    train_flag = False
    cwd = os.getcwd()
    cpu_values_list = None
    memory_values_list = None
    train_cpu_file_name = "train_cpu_quota_runtime_values.csv"
    train_memory_file_name = "train_memory_runtime_values.csv"
    inference_cpu_file_name = "inference_cpu_quota_runtime_values.csv"
    inference_memory_file_name = "inference_memory_runtime_values.csv"
    global model_dict
    dataset_folders = f'{cwd}/host_data'
    dataset_name = questionary.select("Choose dataset:",choices=os.listdir(dataset_folders)).ask()
    model_dict = {"model": None,"dataset": None, "normalizetype": None,"processing_type": None}
    model_folders = os.path.join(dataset_folders,dataset_name)
    model_name = questionary.select("Choose model folder:", choices=os.listdir(model_folders)).ask()
    model_dir = os.path.join(model_folders,model_name)
    plot_selection = questionary.select("Train or inference:", choices=["train", "inference"]).ask()
    if plot_selection == "train":
        train_flag = True
        cpu_values_file = os.path.join(model_dir, train_cpu_file_name)
        memory_values_file = os.path.join(model_dir, train_memory_file_name)    
    else:
        cpu_values_file = os.path.join(model_dir, inference_cpu_file_name)
        memory_values_file = os.path.join(model_dir, inference_memory_file_name)
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

    else: cpu_log_list = None
    if cpu_values_list != None and memory_values_list != None:
        return cpu_values_list, memory_values_list, model_name, dataset_name, train_flag
    elif cpu_values_list != None:
        return cpu_values_list, None, model_name, dataset_name, train_flag
    elif memory_values_list != None:
        return None,memory_values_list, model_name, dataset_name, train_flag
    else: return None,None, model_name, dataset_name


def folder_create(folder_path: str = None):
    isdir = os.path.isdir(folder_path)
    if isdir:
        pass
    else:
        os.mkdir(folder_path)
        
    
def folder_check(folder_path: str = None):
    isdir = os.path.isdir(folder_path)
    if isdir:
        return True
    else:
        return False

def file_check(file_path: str = None):
    isfile = os.path.isfile(file_path)
    if isfile:
        return True
    else:
        return False

def single_plots(x_axis: list = None, x1_axis: list = None, y_axis: list = None, model_name: str = None, dataset_name: str = None, RAM_flag: bool = False, CPU_flag: bool = False, train_flag: bool = False):
    cwd = os.getcwd()
    args_time_list=[]
    init_time_list=[]
    setup_time_list=[]
    eval_time_list=[]
    create_folder_list=[]
    date_now = datetime.date.today()

    plots_folder = os.path.join(cwd,'plots')
    folder_create(plots_folder)
    
    model_folder = os.path.join(plots_folder, model_name)
    folder_create(model_folder)
    
    dataset_folder = os.path.join(model_folder, dataset_name)
    folder_create(dataset_folder)

    date_folder = os.path.join(dataset_folder, str(date_now))
    folder_create(date_folder)


    plots_path=os.path.join(plots_folder, model_name, dataset_name, str(date_now))
    folder_create(plots_path)

    
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
        
        if train_flag:
            ax1.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Train Args Time")
            ax2.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Train Create Folder Time")
            ax3.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Train Init Time")
            ax4.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Train Setup Time")
            ax5.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Train Eval Time")
            ax6.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Train Total Time")
            plots_path = os.path.join(plots_path, "RAM Results")
            folder_create(plots_path)
            ax1.savefig(f'{plots_path}/{date_now}_Train_RAM_1.png')
            ax2.savefig(f'{plots_path}/{date_now}_Train_RAM_2.png')
            ax3.savefig(f'{plots_path}/{date_now}_Train_RAM_3.png')
            ax4.savefig(f'{plots_path}/{date_now}_Train_RAM_4.png')
            ax5.savefig(f'{plots_path}/{date_now}_Train_RAM_5.png')
            ax6.savefig(f'{plots_path}/{date_now}_Train_RAM_6.png')
        else:
            ax1.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Inference Args Time")
            ax2.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Inference Create Folder Time")
            ax3.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Inference Init Time")
            ax4.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Inference Setup Time")
            ax5.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Inference Eval Time")
            ax6.fig.suptitle(f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Inference Total Time")
            plots_path = os.path.join(plots_path, "RAM Results")
            folder_create(plots_path)
            ax1.savefig(f'{plots_path}/{date_now}_Inference_RAM_1.png')
            ax2.savefig(f'{plots_path}/{date_now}_Inference_RAM_2.png')
            ax3.savefig(f'{plots_path}/{date_now}_Inference_RAM_3.png')
            ax4.savefig(f'{plots_path}/{date_now}_Inference_RAM_4.png')
            ax5.savefig(f'{plots_path}/{date_now}_Inference_RAM_5.png')
            ax6.savefig(f'{plots_path}/{date_now}_Inference_RAM_6.png')
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

        if train_flag:
            ax1.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Train Args Time")
            ax2.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Train Create Folder Time")
            ax3.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Train Init Time")
            ax4.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Train Setup Time")
            ax5.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Train Eval Time")
            ax6.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Train Total Time")

            plots_path = os.path.join(plots_path, "CPU Results")
            folder_create(plots_path)

            ax1.savefig(f'{plots_path}/{date_now}_Train_CPU_1.png')
            ax2.savefig(f'{plots_path}/{date_now}_Train_CPU_2.png')
            ax3.savefig(f'{plots_path}/{date_now}_Train_CPU_3.png')
            ax4.savefig(f'{plots_path}/{date_now}_Train_CPU_4.png')
            ax5.savefig(f'{plots_path}/{date_now}_Train_CPU_5.png')
            ax6.savefig(f'{plots_path}/{date_now}_Train_CPU_6.png')
        else:
            ax1.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Inference Args Time")
            ax2.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Inference Create Folder Time")
            ax3.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Inference Init Time")
            ax4.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Inference Setup Time")
            ax5.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Inference Eval Time")
            ax6.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Inference Total Time")

            plots_path = os.path.join(plots_path, "CPU Results")
            folder_create(plots_path)

            ax1.savefig(f'{plots_path}/{date_now}_Inference_CPU_1.png')
            ax2.savefig(f'{plots_path}/{date_now}_Inference_CPU_2.png')
            ax3.savefig(f'{plots_path}/{date_now}_Inference_CPU_3.png')
            ax4.savefig(f'{plots_path}/{date_now}_Inference_CPU_4.png')
            ax5.savefig(f'{plots_path}/{date_now}_Inference_CPU_5.png')
            ax6.savefig(f'{plots_path}/{date_now}_Inference_CPU_6.png')

def epoch_plot (cpu_runtime_values = None, ram_runtime_values = None, model_name: str = None, dataset_name: str = None):    
    cwd = os.getcwd()
    epoch_patterns = ['-----Epoch']
    match_list = []
    tmp_list = []
    plots_folder = os.path.join(cwd,'plots')
    folder_create(plots_folder)
    
    model_folder = os.path.join(plots_folder, model_name)
    folder_create(model_folder)
    
    dataset_folder = os.path.join(model_folder, dataset_name)
    folder_create(dataset_folder)
    date_now = datetime.date.today()
 


    date_folder = os.path.join(dataset_folder, str(date_now))
    folder_create(date_folder)

    plots_path = os.path.join(date_folder, 'epoch_time_plot')
    folder_create(plots_path)

    date_now = datetime.date.today()
    logs_dir = f'{cwd}/host_data/{dataset_name}/{model_name}/'
    
    RAM_log_path = os.path.join (logs_dir,'RAM_log')
    CPU_log_path = os.path.join(logs_dir, 'CPU_log')
    RAM_log_flag = folder_check(RAM_log_path)
    CPU_log_flag = folder_check(CPU_log_path)
    # RAM_folders = [folders for folders in RAM_folders if not folders.endswith('.csv')]

    quota_epoch_list = []
    cpu_log_dict = {}
    ram_log_dict = {}
    big_boy_epoch_list_cpu = []
    if RAM_log_flag:
        RAM_folders = os.listdir(RAM_log_path)
        RAM_folders.sort()
        x_axis = []
        y_axis = []
        for folder in RAM_folders:
            folder_path = os.path.join(RAM_log_path, folder)


            last_epoch_timedate_struct = []
            for file in os.listdir(folder_path):
                ram_log_list = []
                ram_rt_list = []
                ram_rt_datetime_struct = []
                match_list =[]
                train_log_timedate_list = []
                train_log_timedate_struct = []
                list_items = None
                list_element = None
                _item = None
                log_file = os.path.join(folder_path, file)
                with open(log_file, "r") as csvfile:
                    csvreader = csv.reader(csvfile, delimiter=',')
                    # runtime_header = next(csvreader, None)
                    for row in csvreader:
                        ram_log_list.append(row)
                _date = ram_log_list[0][0]
                train_log_timedate_list= ram_log_list
                for pattern in epoch_patterns:
                    for list_items in train_log_timedate_list:
                        # print('Looking for %s in "%s" ->' %(pattern,list_items))
                        for text in list_items:
                            if re.search(pattern, text):
                                # print (f"Found match!\n{list_items}")
                                match_list.append(text)
                final_epoch_string = match_list[-1][12:]
                tmp_var = final_epoch_string.split("/")
                last_epoch = int(re.sub("[^0-9]", "", tmp_var[0]))
                for list_element in match_list:
                    # element_time_stamp = list_element[:12]
                    epoch_string = list_element[12:]
                    tmp_var = epoch_string.split("/")
                    current_epoch = int(re.sub("[^0-9]", "", tmp_var[0]))
                    if current_epoch != 0:
                            previous_time_stamp = current_time_stamp
                            current_time_stamp = list_element[:12]
                            last_epoch_time = time.strptime(previous_time_stamp, "%H:%M:%S.%f")
                            current_epoch_time = time.strptime(current_time_stamp, "%H:%M:%S.%f")
                            epoch_train_time = (time.mktime(current_epoch_time) - time.mktime(last_epoch_time))/60
                            # print (epoch_train_time)
                            x_axis.append(epoch_train_time)
                    else:
                        # print("\n")
                        current_time_stamp = list_element[:12]

                    tmp_str = f'{_date} {list_element[:12]}'
                    train_log_timedate_struct.append(time.strptime(tmp_str, "%Y-%m-%d %H:%M:%S.%f"))
                for item in ram_runtime_values:
                    ram_rt_list.append(item[:1])
                    ram_rt_datetime_struct.append(time.strptime(item[0], "%Y-%m-%d %H:%M:%S.%f"))
                i = bisect.bisect_left(ram_rt_datetime_struct, train_log_timedate_struct[-1])
                runtime_ram = ram_runtime_values[i-1][1]
                for index in range(last_epoch):
                    y_axis.append(runtime_ram)

        epoch_df = pd.DataFrame(list(zip(y_axis, x_axis)), columns = ['RAM (MB)', 'Epoch Time'])

        ax1 = sns.relplot(x="RAM (MB)", y="Epoch Time", data=epoch_df, kind='line', ci=90,markers=True, dashes=False)  
        ax1.fig.suptitle(f"{model_name}/{dataset_name}\nRAM (MB) vs Epoch Time (Minutes)")

        ax1.savefig(f'{plots_path}/{date_now}_RAM_epoch.png')
    # else: ram_log_list = None
    
    if CPU_log_flag:
        CPU_folders = os.listdir(CPU_log_path)
        x_axis = []
        y_axis = []
        for folder in CPU_folders:
            folder_path = os.path.join(CPU_log_path, folder)


            last_epoch_timedate_struct = []
            onlyfiles = [f for f in os.listdir(CPU_log_path) if os.path.isfile(os.path.join(CPU_log_path, f))]
            for file in os.listdir(folder_path):
                cpu_log_list = []

                cpu_rt_list = []
                cpu_rt_datetime_struct = []
                match_list =[]
                train_log_timedate_list = []
                train_log_timedate_struct = []
                list_items = None
                list_element = None
                _item = None
                log_file = os.path.join(folder_path, file)
                with open(log_file, "r") as csvfile:
                    csvreader = csv.reader(csvfile, delimiter=',')
                    # runtime_header = next(csvreader, None)
                    for row in csvreader:
                        cpu_log_list.append(row)
                _date = cpu_log_list[0][0]
                train_log_timedate_list= cpu_log_list
                for pattern in epoch_patterns:
                    for list_items in train_log_timedate_list:
                        # print('Looking for %s in "%s" ->' %(pattern,list_items))
                        for text in list_items:
                            if re.search(pattern, text):
                                # print (f"Found match!\n{list_items}")
                                match_list.append(text)
                final_epoch_string = match_list[-1][12:]
                tmp_var = final_epoch_string.split("/")
                last_epoch = int(re.sub("[^0-9]", "", tmp_var[0]))
                for list_element in match_list:
                    # element_time_stamp = list_element[:12]
                    epoch_string = list_element[12:]
                    tmp_var = epoch_string.split("/")
                    current_epoch = int(re.sub("[^0-9]", "", tmp_var[0]))
                    if current_epoch != 0:
                            previous_time_stamp = current_time_stamp
                            current_time_stamp = list_element[:12]
                            last_epoch_time = time.strptime(previous_time_stamp, "%H:%M:%S.%f")
                            current_epoch_time = time.strptime(current_time_stamp, "%H:%M:%S.%f")
                            epoch_train_time = (time.mktime(current_epoch_time) - time.mktime(last_epoch_time))/60
                            # print (epoch_train_time)
                            x_axis.append(epoch_train_time)
                    else:
                        # print("\n")
                        current_time_stamp = list_element[:12]

                    tmp_str = f'{_date} {list_element[:12]}'
                    train_log_timedate_struct.append(time.strptime(tmp_str, "%Y-%m-%d %H:%M:%S.%f"))
                for item in cpu_runtime_values:
                    cpu_rt_list.append(item[:1])
                    cpu_rt_datetime_struct.append(time.strptime(item[0], "%Y-%m-%d %H:%M:%S.%f"))
                i = bisect.bisect_left(cpu_rt_datetime_struct, train_log_timedate_struct[-1])
                train_run_quota = cpu_runtime_values[i-1][1]
                qttc = int(train_run_quota)/100000.0
                for index in range(last_epoch):
                    y_axis.append(qttc)

        epoch_df = pd.DataFrame(list(zip(y_axis, x_axis)), columns = ['CPU (Cores)', 'Epoch Time'])

        ax1 = sns.relplot(x="CPU (Cores)", y="Epoch Time", data=epoch_df, kind='line', ci=90,markers=True, dashes=False)
        ax1.fig.suptitle(f"{model_name}/{dataset_name}\nCPU Cores vs Epoch Time (Minutes)")

        ax1.savefig(f'{plots_path}/{date_now}_CPU_epoch.png')

            # elif max_epoch_counter == 0:
        #     raise ValueError("Could not calculate epoch!")
    # if CPU_flag:
    print ("Epoch plot done!")


if __name__=="__main__":
    train_flag = False
    cpu_values_list = None
    memory_values_list = None
    cpu_values_list, memory_values_list, model_name, dataset_name, train_flag = init()
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
        single_plots(function_time_list, total_time_list, RAM_limit_list, model_name, dataset_name, RAM_flag, CPU_flag, train_flag)    
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
                function_time_list.append(float(function_time))
                time_sum+=float(function_time)
            total_time_list.append(time_sum)
            time_sum=0.0
        single_plots(function_time_list, total_time_list, CPU_limit_list, model_name, dataset_name, RAM_flag, CPU_flag, train_flag)
    else: print("No CPU file found!")
    if train_flag:
        epoch_plot(cpu_runtime_values=cpu_values_list, ram_runtime_values=memory_values_list, model_name=model_name, dataset_name=dataset_name)
    print ("Script ran sucessfully!")
