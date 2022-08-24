import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import questionary
import os
import csv
import re
import bisect

##### Lists ##### 
model_name_list = ['MLP', 'Resnet1d', 'Alexnet1d', 'LeNet1d', 'CNN_1d', 'BiLSTM1d']
dataset_name_list = ['SEU', 'MFPT', 'CWRU']

##### File names #####
train_cpu_file_name = 'train_cpu_quota_runtime_values.csv'
train_memory_file_name = 'train_memory_runtime_values.csv'
inference_cpu_file_name = 'inference_cpu_quota_runtime_values.csv'
inference_memory_file_name = 'inference_memory_runtime_values.csv'
##### Paths #####
host_data_path = os.path.join(os.getcwd(), 'host_data') # host machine path of shared volume
plots_path = os.path.join(os.getcwd(), 'plots') # host machine path for saving plots

#### Others #####
timedate_length = 23 # Format: YYYY-MM-DD hh:mm:ss.%f, for example 2022-06-26 21:09:46.010881
lowest_model_memory_run = 10000 # given in MB
epoch_begin_pattern = '-----Epoch'
train_log_file_pattern = 'training.log'
cwd = os.getcwd()
CNN_name = 'CNN1d'
alexnet_name = 'AlexNet1d'
resnet_name = 'ResNet1d'

def folder_create(folder_path: str = None):
    """Checks if folder_path exists. If not, it creates it.

    Args:
        folder_path (str): path to check/create folder. Defaults to None.
    """
    isdir = os.path.isdir(folder_path)
    if isdir:
        pass
    else:
        os.mkdir(folder_path)
        
def folder_check(folder_path: str = None):
    """Checks if folder_path exists. Returns boolean value of evaluation

    Args:
        folder_path (str): path to check. Defaults to None.
    
    Returns:
        True/False: True if folder exists, else false
    """
    isdir = os.path.isdir(folder_path)
    if isdir:
        return True
    else:
        return False

def file_check(file_path: str = None):
    """Checks if file_path exists. Returns boolean value of evaluation

    Args:
        file_path (str): path to check. Defaults to None.
    
    Returns:
        True/False: True if file exists, else false
    """
    isfile = os.path.isfile(file_path)
    if isfile:
        return True
    else:
        return False

def plot_init():
    sns.set_theme(style='white', font_scale=1.5, color_codes=True, rc={"figure.dpi":100, 'savefig.dpi':1000, "figure.figsize":(12, 6)})
    # sns.set_style("white")
    # sns.axes_style("white")
    # sns.set(font_scale = 1.5, )
    # sns.color_palette()
def single_plots(x_axis: list = None, x1_axis: list = None, y_axis: list = None, model_name: str = None, dataset_name: str = None, RAM_flag: bool = False, CPU_flag: bool = False, train_flag: bool = False):
    """_summary_

    Args:
        x_axis (list): Contains each function execution times as a list in this order -> args, init, setup, eval . Defaults to None.
        x1_axis (list): Total execution time for each run. Defaults to None.
        y_axis (list): RAM or CPU quota value for each run. Defaults to None.
        model_name (str)
        dataset_name (str)
        RAM_flag (bool): If this is active then CPU_flag must be false. Defaults to False.
        CPU_flag (bool): If this is active then RAM_flag must be false. Defaults to False.
        train_flag (bool): If it is false, then inference plots will be generated. If True, train plots will be generated. Defaults to False.
    """
    cwd = os.getcwd()
    args_time_list=[]
    init_time_list=[]
    setup_time_list=[]
    eval_time_list=[]
    create_folder_list=[]
    date_now = datetime.date.today()

    plots_folder = os.path.join(cwd,'plots')
    
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

        ax1 = sns.relplot(x="RAM (MB)", y="Args Time", data=args_df, kind='line', ci=90,markers=True, dashes=False, height=7, aspect=5/4)
        ax2 = sns.relplot(x="RAM (MB)", y="Create Folder Time", kind="line", data=folder_df, ci=90,markers=True, dashes=False, height=7, aspect=5/4)
        ax3 = sns.relplot(x="RAM (MB)", y="Init Time", kind="line", data=init_df, ci=90,markers=True, dashes=False, height=7, aspect=5/4)
        ax4 = sns.relplot(x="RAM (MB)", y="Setup Time", kind="line", data=setup_df, ci=90, markers=True, dashes=False, height=7, aspect=5/4)
        ax5 = sns.relplot(x="RAM (MB)", y="Eval Time", kind="line", data=eval_df, ci=90, markers=True, dashes=False, height=7, aspect=5/4)
        ax6 = sns.relplot(x="RAM (MB)", y="Total Time", kind="line", data=total_df, ci=90, markers=True, dashes=False, height=7, aspect=5/4)        
        plots_path = os.path.join(plots_path, "RAM Results")
        folder_create(plots_path)
        if train_flag:
            save_fig(ax1, f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Train Args Time", f'{plots_path}/{date_now}_Train_RAM_1.svg')
            save_fig(ax2, f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Train Create Folder Time", f'{plots_path}/{date_now}_Train_RAM_2.svg')
            save_fig(ax3, f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Train Init Time", f'{plots_path}/{date_now}_Train_RAM_3.svg')
            save_fig(ax4, f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Train Init Time", f'{plots_path}/{date_now}_Train_RAM_4.svg')
            save_fig(ax5, f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Train Setup Time", f'{plots_path}/{date_now}_Train_RAM_5.svg')
            save_fig(ax6, f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Train Init Time", f'{plots_path}/{date_now}_Train_RAM_6.svg')
        else:
            save_fig(ax1, f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Inference Args Time", f'{plots_path}/{date_now}_Inference_RAM_1.svg')
            save_fig(ax2, f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Inference Create Folder Time", f'{plots_path}/{date_now}_Inference_RAM_2.svg')
            save_fig(ax3, f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Inference Init Time", f'{plots_path}/{date_now}_Inference_RAM_3.svg')
            save_fig(ax4, f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Inference Init Time", f'{plots_path}/{date_now}_Inference_RAM_4.svg')
            save_fig(ax5, f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Inference Setup Time", f'{plots_path}/{date_now}_Inference_RAM_5.svg')
            save_fig(ax6, f"Model: {model_name}/Dataset: {dataset_name}\nRAM vs Inference Init Time", f'{plots_path}/{date_now}_Inference_RAM_6.svg')

    if CPU_flag:
        args_df = pd.DataFrame(list(zip(y_axis, args_time_list)), columns = ['CPU Cores', 'Args Time'])
        folder_df = pd.DataFrame(list(zip(y_axis, create_folder_list)), columns = ['CPU Cores', 'Create Folder Time'])
        init_df = pd.DataFrame(list(zip(y_axis, init_time_list)), columns = ['CPU Cores', 'Init Time'])
        setup_df = pd.DataFrame(list(zip(y_axis, setup_time_list)), columns = ['CPU Cores', 'Setup Time'])
        eval_df = pd.DataFrame(list(zip(y_axis, eval_time_list)), columns = ['CPU Cores', 'Eval Time'])
        total_df = pd.DataFrame(list(zip(y_axis, x1_axis)), columns = ['CPU Cores', 'Total Time'])

        ax1 = sns.relplot(x="CPU Cores", y="Args Time", data=args_df, kind='line', ci=90,markers=True, dashes=False, height=7, aspect=5/4)
        ax2 = sns.relplot(x="CPU Cores", y="Create Folder Time", kind="line", data=folder_df, ci=90,markers=True, dashes=False, height=7, aspect=5/4)
        ax3 = sns.relplot(x="CPU Cores", y="Init Time", kind="line", data=init_df, ci=90,markers=True, dashes=False, height=7, aspect=5/4)
        ax4 = sns.relplot(x="CPU Cores", y="Setup Time", kind="line", data=setup_df, ci=90, markers=True, dashes=False, height=7, aspect=5/4)
        ax5 = sns.relplot(x="CPU Cores", y="Eval Time", kind="line", data=eval_df, ci=90, markers=True, dashes=False, height=7, aspect=5/4)
        ax6 = sns.relplot(x="CPU Cores", y="Total Time", kind="line", data=total_df, ci=90, markers=True, dashes=False, height=7, aspect=5/4)
        plots_path = os.path.join(plots_path, "CPU Results")
        folder_create(plots_path)

        if train_flag:
            save_fig(ax1, f"Model: {model_name}/Dataset: {dataset_name}\nCPU Cores vs Train Args Time", f'{plots_path}/{date_now}_Train_CPU_1.svg')
            save_fig(ax2, f"Model: {model_name}/Dataset: {dataset_name}\nCPU Cores vs Train Create Folder Time", f'{plots_path}/{date_now}_Train_CPU_2.svg')
            save_fig(ax3, f"Model: {model_name}/Dataset: {dataset_name}\nCPU Cores vs Train Init Time", f'{plots_path}/{date_now}_Train_CPU_3.svg')
            save_fig(ax4, f"Model: {model_name}/Dataset: {dataset_name}\nCPU Cores vs Train Init Time", f'{plots_path}/{date_now}_Train_CPU_4.svg')
            save_fig(ax5, f"Model: {model_name}/Dataset: {dataset_name}\nCPU Cores vs Train Setup Time", f'{plots_path}/{date_now}_Train_CPU_5.svg')
            
        else:
            save_fig(ax1, f"Model: {model_name}/Dataset: {dataset_name}\nCPU Cores vs Inference Args Time", f'{plots_path}/{date_now}_Inference_CPU_1.svg')
            save_fig(ax2, f"Model: {model_name}/Dataset: {dataset_name}\nCPU Cores vs Inference Create Folder Time", f'{plots_path}/{date_now}_Inference_CPU_2.svg')
            save_fig(ax3, f"Model: {model_name}/Dataset: {dataset_name}\nCPU Cores vs Inference Init Time", f'{plots_path}/{date_now}_Inference_CPU_3.svg')
            save_fig(ax4, f"Model: {model_name}/Dataset: {dataset_name}\nCPU Cores vs Inference Init Time", f'{plots_path}/{date_now}_Inference_CPU_4.svg')
            save_fig(ax5, f"Model: {model_name}/Dataset: {dataset_name}\nCPU Cores vs Inference Setup Time", f'{plots_path}/{date_now}_Inference_CPU_5.svg')
            save_fig(ax6, f"Model: {model_name}/Dataset: {dataset_name}\nCPU Cores vs Inference Init Time", f'{plots_path}/{date_now}_Inference_CPU_6.svg')


def epoch_plot (cpu_runtime_values = None, ram_runtime_values = None, model_name: str = None, dataset_name: str = None):    
    cwd = os.getcwd()
    epoch_patterns = ['-----Epoch']
    match_list = []
    tmp_list = []
    plots_folder = os.path.join(cwd,'plots')
    
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

    # if RAM_log_flag:
    #     RAM_folders = os.listdir(RAM_log_path)
    #     RAM_folders.sort()
    #     x_axis = []
    #     y_axis = []
        
    #     for folder in RAM_folders:
    #         folder_path = os.path.join(RAM_log_path, folder)

    #         for file in os.listdir(folder_path):
    #             for pattern in ['training.log']:
    #                     # print('Looking for %s in "%s" ->' %(pattern,list_items))
    #                         if re.search(pattern, str(file)):
    #                             # print (f"Found match!\n{list_items}")
    #                             ram_log_list = []
    #                             ram_rt_list = []
    #                             ram_rt_datetime_struct = []
    #                             match_list =[]
    #                             train_log_timedate_list = []
    #                             train_log_timedate_struct = []
    #                             list_items = None
    #                             list_element = None
    #                             _item = None
    #                             for item in ram_runtime_values:
    #                                 ram_rt_list.append(item[:1])
    #                                 ram_rt_datetime_struct.append(datetime.datetime.strptime(item[0], "%Y-%m-%d %H:%M:%S.%f"))
    #                             log_file = os.path.join(folder_path, file)
    #                             with open(log_file, "r") as csvfile:
    #                                 csvreader = csv.reader(csvfile, delimiter=',')
    #                                 # runtime_header = next(csvreader, None)
    #                                 for row in csvreader:
    #                                     ram_log_list.append(row)
    #                             _date = ram_log_list[0][0]
    #                             train_log_timedate_list= ram_log_list
    #                             for pattern in epoch_patterns:
    #                                 for list_items in train_log_timedate_list:
    #                                     # print('Looking for %s in "%s" ->' %(pattern,list_items))
    #                                     for text in list_items:
    #                                         if re.search(pattern, text):
    #                                             # print (f"Found match!\n{list_items}")
    #                                             _match_var = _date + " " + text
    #                                             match_list.append(_match_var)
    #                             final_epoch_string = match_list[-1][timedate_length:]
    #                             tmp_var = final_epoch_string.split("/")
    #                             last_epoch = int(re.sub("[^0-9]", "", tmp_var[0]))
    #                             for list_element in match_list:
    #                                 # element_time_stamp = list_element[:12]
    #                                 epoch_string = list_element[timedate_length:]
    #                                 tmp_var = epoch_string.split("/")
    #                                 current_epoch = int(re.sub("[^0-9]", "", tmp_var[0]))
    #                                 if current_epoch != 0:
    #                                         previous_time_stamp = current_time_stamp
    #                                         current_time_stamp = list_element[:timedate_length]
    #                                         last_epoch_time = datetime.datetime.strptime(previous_time_stamp, "%Y-%m-%d %H:%M:%S.%f")
    #                                         current_epoch_time = datetime.datetime.strptime(current_time_stamp, "%Y-%m-%d %H:%M:%S.%f")
    #                                         difference = current_epoch_time - last_epoch_time
    #                                         days = difference.days
    #                                         hours, remainder = divmod(difference.seconds, 3600)
    #                                         minutes, seconds = divmod(remainder, 60)
    #                                         epoch_train_time = days*24*60*60 + hours*60*60 + minutes*60 + seconds
    #                                         train_log_timedate_struct.append(datetime.datetime.strptime(current_time_stamp, "%Y-%m-%d %H:%M:%S.%f"))
    #                                         x_axis.append(epoch_train_time)
    #                                 else:
    #                                     # print("\n")
    #                                     current_time_stamp = list_element[:timedate_length]
    #                                     train_log_timedate_struct.append(datetime.datetime.strptime(current_time_stamp, "%Y-%m-%d %H:%M:%S.%f"))

    #                             i = bisect.bisect_left(ram_rt_datetime_struct, train_log_timedate_struct[-1])
    #                             runtime_ram = ram_runtime_values[i-1][1]
    #                             for index in range(last_epoch):
    #                                 y_axis.append(int(runtime_ram))

    #     epoch_df = pd.DataFrame(list(zip(y_axis, x_axis)), columns = ['RAM (MB)', 'Epoch Time'])
    #     # removing negative time differences, resulting from date changes
    #     epoch_df[epoch_df < 0] = 0
    #     ax1 = sns.relplot(x="RAM (MB)", y="Epoch Time", data=epoch_df, kind='line', ci=90,markers=True, dashes=False, height=7, aspect=5/4)  
    #     ax1.fig.suptitle(f"{model_name}/{dataset_name}\nRAM (MB) vs Epoch Time (s)",y=1.08)

    #     ax1.savefig(f'{plots_path}/{date_now}_RAM_epoch.svg')
    # else: ram_log_list = None
    
    if CPU_log_flag:
        CPU_folders = os.listdir(CPU_log_path)
        x_axis = []
        y_axis = []
        for folder in CPU_folders:
            folder_path = os.path.join(CPU_log_path, folder)
            for file in os.listdir(folder_path):
                for pattern in ['training.log']:
                        # print('Looking for %s in "%s" ->' %(pattern,list_items))
                            if re.search(pattern, str(file)):
                                # print (f"Found match!\n{list_items}")
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
                                for item in cpu_runtime_values:
                                    cpu_rt_list.append(item[:1])
                                    cpu_rt_datetime_struct.append(datetime.datetime.strptime(item[0], "%Y-%m-%d %H:%M:%S.%f"))
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
                                                _match_var = _date + " " + text
                                                match_list.append(_match_var)
                                final_epoch_string = match_list[-1][timedate_length:]
                                tmp_var = final_epoch_string.split("/")
                                last_epoch = int(re.sub("[^0-9]", "", tmp_var[0]))
                                for list_element in match_list:
                                    # element_time_stamp = list_element[:12]
                                    epoch_string = list_element[timedate_length:]
                                    tmp_var = epoch_string.split("/")
                                    current_epoch = int(re.sub("[^0-9]", "", tmp_var[0]))
                                    if current_epoch != 0:
                                            previous_time_stamp = current_time_stamp
                                            current_time_stamp = list_element[:timedate_length]
                                            last_epoch_time = datetime.datetime.strptime(previous_time_stamp, "%Y-%m-%d %H:%M:%S.%f")
                                            current_epoch_time = datetime.datetime.strptime(current_time_stamp, "%Y-%m-%d %H:%M:%S.%f")
                                            difference = current_epoch_time - last_epoch_time
                                            days = difference.days
                                            hours, remainder = divmod(difference.seconds, 3600)
                                            minutes, seconds = divmod(remainder, 60)
                                            epoch_train_time = days*24*60*60 + hours*60*60 + minutes*60 + seconds
                                            train_log_timedate_struct.append(datetime.datetime.strptime(current_time_stamp, "%Y-%m-%d %H:%M:%S.%f"))
                                            # print (epoch_train_time)
                                            x_axis.append(epoch_train_time)
                                    else:
                                        # print("\n")
                                        current_time_stamp = list_element[:timedate_length]
                                        train_log_timedate_struct.append(datetime.datetime.strptime(current_time_stamp, "%Y-%m-%d %H:%M:%S.%f"))
                                i = bisect.bisect_left(cpu_rt_datetime_struct, train_log_timedate_struct[-1])
                                train_run_quota = cpu_runtime_values[i-1][1]
                                qttc = int(train_run_quota)/100000.0
                                for index in range(last_epoch):
                                    y_axis.append(qttc)

        epoch_df = pd.DataFrame(list(zip(y_axis, x_axis)), columns = ['CPU (Cores)', 'Epoch Time'])
        # removing negative time differences, resulting from date changes
        epoch_df[epoch_df < 0] = 0
        ax1 = sns.relplot(x="CPU (Cores)", y="Epoch Time", data=epoch_df, kind='line', ci=90,markers=True, dashes=False, height=7, aspect=5/4)
        save_fig(ax1, f"{model_name}/{dataset_name}\nCPU Cores vs Epoch Time (s)", f'{plots_path}/{date_now}_CPU_epoch.svg')

    print ("Epoch plot done!")

def plot_all_models(dir: str = os.getcwd(), cpu_limit: int = None):
    """CURRENTLY NOT WORKING: Generates combined training plot of all available models for each individual dataset.

    Args:
        dir (str): Path to logs directory. Defaults to cwd.
        cpu_limit (int): Top limit for CPU plotting (e.g 1 core). Defaults to None.
    """
    cwd = os.getcwd()
    host_data_path = os.path.join(cwd,'host_data')
    dataset_folder_list = ['SEU', 'MFPT']
    model_folder_list = ['MLP', 'Alexnet1d', 'CNN_1d', 'Resnet1d', 'LeNet1d', 'BiLSTM1d']                                                                                                  
    ram_big_boy_list = []
    r = []
    y_axis = []
    x_axis_ram = []
    cpu_values_list = []
    memory_values_list = []
    train_cpu_file_name = "train_cpu_quota_runtime_values.csv"
    train_memory_file_name = "train_memory_runtime_values.csv"
    # model_list = questionary.checkbox("Choose models to overlay:", choices=model_folder_list).ask()
    timedate_length = 23                                                                                                  
    subdirs = [x[0] for x in os.walk(dir)]
    lowest_ram_list = []
    for dataset in dataset_folder_list:
        print(f"At {dataset}\n")
        ram_big_boy_list = []
        cpu_big_boy_list = []
        for model in model_name_list:
            x_axis_cpu = []
            x_axis_ram = []
            lowest_model_memory_run = 10000
            for subdir in subdirs:
                RAM_folder_pattern = os.path.join(host_data_path, dataset, model, 'RAM_log')
                CPU_folder_pattern = os.path.join(host_data_path,dataset,model,'CPU_log')
                if re.search(RAM_folder_pattern, subdir):
                    memory_values_list = []
                    _tmp_file_path = os.path.join(host_data_path,dataset,model,train_memory_file_name)
                    with open(_tmp_file_path, "r") as csvfile:
                        csvreader = csv.reader(csvfile, delimiter=',')
                        runtime_header = next(csvreader, None)
                        for row in csvreader:
                            memory_values_list.append(row)
                    for value in memory_values_list:
                            if int(value[1]) < lowest_model_memory_run:
                                lowest_model_memory_run = int(value[1])
                    ram_log_files = os.walk(subdir).__next__()[2]
                    for file in ram_log_files:
                        if re.search('training.log', file):
                            ram_log_list = []
                            ram_rt_list = []
                            ram_rt_datetime_struct = []
                            match_list =[]
                            train_log_timedate_list = []
                            train_log_timedate_struct = []
                            list_items = None
                            list_element = None
                            _item = None
                            for item in memory_values_list:
                                ram_rt_list.append(item[:1])
                                ram_rt_datetime_struct.append(datetime.datetime.strptime(item[0], "%Y-%m-%d %H:%M:%S.%f"))
                            RAM_log_file_path = os.path.join(subdir, file)
                            with open(RAM_log_file_path, "r") as csvfile:
                                csvreader = csv.reader(csvfile, delimiter=',')
                                # runtime_header = next(csvreader, None)
                                for row in csvreader:
                                    ram_log_list.append(row)
                            _date = ram_log_list[0][0]
                            train_log_timedate_list= ram_log_list
                            for list_items in train_log_timedate_list:
                                # print('Looking for %s in "%s" ->' %(pattern,list_items))
                                for text in list_items:
                                    if re.search('-----Epoch', text):
                                        # print (f"Found match!\n{list_items}")
                                        _match_var = _date + " " + text
                                        match_list.append(_match_var)
                            final_epoch_string = match_list[-1][timedate_length:]
                            tmp_var = final_epoch_string.split("/")
                            last_epoch = int(re.sub("[^0-9]", "", tmp_var[0]))
                            for list_element in match_list:
                                # element_time_stamp = list_element[:12]
                                epoch_string = list_element[timedate_length:]
                                tmp_var = epoch_string.split("/")
                                current_epoch = int(re.sub("[^0-9]", "", tmp_var[0]))
                                if current_epoch != 0:
                                        previous_time_stamp = current_time_stamp
                                        current_time_stamp = list_element[:timedate_length]
                                        last_epoch_time = datetime.datetime.strptime(previous_time_stamp, "%Y-%m-%d %H:%M:%S.%f")
                                        current_epoch_time = datetime.datetime.strptime(current_time_stamp, "%Y-%m-%d %H:%M:%S.%f")
                                        difference = current_epoch_time - last_epoch_time
                                        days = difference.days
                                        hours, remainder = divmod(difference.seconds, 3600)
                                        minutes, seconds = divmod(remainder, 60)
                                        epoch_train_time = days*24*60*60 + hours*60*60 + minutes*60 + seconds
                                        train_log_timedate_struct.append(datetime.datetime.strptime(current_time_stamp, "%Y-%m-%d %H:%M:%S.%f"))
                                        # big_boy_list.append([dataset, model, epoch_train_time, runtime_ram])
                                        if epoch_train_time > 0:
                                            x_axis_ram.append(epoch_train_time)
                                else:
                                    # print("\n")
                                    current_time_stamp = list_element[:timedate_length]
                                    train_log_timedate_struct.append(datetime.datetime.strptime(current_time_stamp, "%Y-%m-%d %H:%M:%S.%f"))
                            i = bisect.bisect_left(ram_rt_datetime_struct, train_log_timedate_struct[-1])
                            runtime_ram = int(memory_values_list[i-1][1])

                            for item in x_axis_ram:
                                ram_big_boy_list.append([model, item, runtime_ram])
                
                if re.search(CPU_folder_pattern, subdir):
                    cpu_values_list = []
                    _tmp_file_path = os.path.join(host_data_path,dataset,model,train_cpu_file_name)
                    with open(_tmp_file_path, "r") as csvfile:
                        csvreader = csv.reader(csvfile, delimiter=',')
                        runtime_header = next(csvreader, None)
                        for row in csvreader:
                            cpu_values_list.append(row)
                    cpu_log_files = os.walk(subdir).__next__()[2]
                    for file in cpu_log_files:
                        if re.search('training.log', file):
                            cpu_log_list = []
                            cpu_rt_list = []
                            cpu_rt_datetime_struct = []
                            match_list =[]
                            train_log_timedate_list = []
                            train_log_timedate_struct = []
                            list_items = None
                            list_element = None
                            _item = None
                            for item in cpu_values_list:
                                cpu_rt_list.append(item[:1])
                                cpu_rt_datetime_struct.append(datetime.datetime.strptime(item[0], "%Y-%m-%d %H:%M:%S.%f"))
                            CPU_log_file_path = os.path.join(subdir, file)
                            with open(CPU_log_file_path, "r") as csvfile:
                                csvreader = csv.reader(csvfile, delimiter=',')
                                # runtime_header = next(csvreader, None)
                                for row in csvreader:
                                    cpu_log_list.append(row)
                            _date = cpu_log_list[0][0]
                            train_log_timedate_list = cpu_log_list
                            for list_items in train_log_timedate_list:
                                # print('Looking for %s in "%s" ->' %(pattern,list_items))
                                for text in list_items:
                                    if re.search('-----Epoch', text):
                                        # print (f"Found match!\n{list_items}")
                                        _match_var = _date + " " + text
                                        match_list.append(_match_var)
                            final_epoch_string = match_list[-1][timedate_length:]
                            tmp_var = final_epoch_string.split("/")
                            last_epoch = int(re.sub("[^0-9]", "", tmp_var[0]))
                            for list_element in match_list:
                                # element_time_stamp = list_element[:12]
                                epoch_string = list_element[timedate_length:]
                                tmp_var = epoch_string.split("/")
                                current_epoch = int(re.sub("[^0-9]", "", tmp_var[0]))
                                if current_epoch != 0:
                                        previous_time_stamp = current_time_stamp
                                        current_time_stamp = list_element[:timedate_length]
                                        last_epoch_time = datetime.datetime.strptime(previous_time_stamp, "%Y-%m-%d %H:%M:%S.%f")
                                        current_epoch_time = datetime.datetime.strptime(current_time_stamp, "%Y-%m-%d %H:%M:%S.%f")
                                        difference = current_epoch_time - last_epoch_time
                                        days = difference.days
                                        hours, remainder = divmod(difference.seconds, 3600)
                                        minutes, seconds = divmod(remainder, 60)
                                        epoch_train_time = days*24*60*60 + hours*60*60 + minutes*60 + seconds
                                        train_log_timedate_struct.append(datetime.datetime.strptime(current_time_stamp, "%Y-%m-%d %H:%M:%S.%f"))
                                        # big_boy_list.append([dataset, model, epoch_train_time, runtime_ram])
                                        if epoch_train_time > 0:
                                            x_axis_cpu.append(epoch_train_time)
                                else:
                                    # print("\n")
                                    current_time_stamp = list_element[:timedate_length]
                                    train_log_timedate_struct.append(datetime.datetime.strptime(current_time_stamp, "%Y-%m-%d %H:%M:%S.%f"))
                            i = bisect.bisect_left(cpu_rt_datetime_struct, train_log_timedate_struct[0])
                            train_run_quota = cpu_values_list[i-1][1]
                            CPU_qtc = int(train_run_quota)/100000.0 #qtc = quota to cores
                            if cpu_limit == None or cpu_limit > CPU_qtc:
                                for item in x_axis_cpu:
                                    if model == 'CNN_1d':
                                        cpu_big_boy_list.append([CNN_name, item, CPU_qtc])
                                    elif model == 'Alexnet1d':
                                        cpu_big_boy_list.append([alexnet_name, item, CPU_qtc])
                                    elif model == 'Resnet1d':
                                        cpu_big_boy_list.append([resnet_name, item, CPU_qtc])
                                    else:
                                        cpu_big_boy_list.append([model, item, CPU_qtc])
            if lowest_model_memory_run != 10000:
                if model == 'CNN_1d':
                    lowest_ram_list.append([dataset, CNN_name, lowest_model_memory_run])
                elif model == 'Alexnet1d':
                    lowest_ram_list.append([dataset, alexnet_name, lowest_model_memory_run])
                elif model == 'Resnet1d':
                    lowest_ram_list.append([dataset, resnet_name, lowest_model_memory_run])
                else:
                    lowest_ram_list.append([dataset, model, lowest_model_memory_run])
        cpu_big_df = pd.DataFrame(cpu_big_boy_list, columns=['Model', 'Time', 'CPU Cores'])
        # ram_big_boy_df = pd.DataFrame(ram_big_boy_list, columns=['Model', 'Time', 'RAM(MB)'])
        cpu_big_df = remove_outlier(cpu_big_df,'Time')
        # ram_big_boy_df = remove_outlier(ram_big_boy_df, 'Time')

        ax2 = sns.relplot(data=cpu_big_df, x='CPU Cores', y='Time', hue='Model', kind='line', height=7, aspect=5/4)
        save_fig(ax2, f"{dataset}\nCPU Cores vs Epoch Time (seconds))", f'{cwd}/plots/{dataset}_CPU_epoch_all_models.svg')
        plt.figure()
    low_ram_df = pd.DataFrame(lowest_ram_list, columns=['Dataset', 'Model', 'Minimum RAM'])
    ax3 = sns.barplot(data=low_ram_df, x='Model', y='Minimum RAM', hue='Dataset').set_title(f"Dataset:Minimum RAM required for training")
    fig = ax3.get_figure()
    fig.savefig(f'{cwd}/plots/RAM_bar_plot_train.svg')
    

def inference_plot(dir, cpu_limit: int = None):
    """Generate combined inference plot of all available models for each individual dataset

    Args:
        dir (str): Path to logs directory. Defaults to cwd.
        cpu_limit (int): Top limit for CPU plotting (e.g 1 core). Defaults to None.
    """
    cwd = os.getcwd()                                                                                               
    ram_big_list = []
    cpu_big_list = []
    cpu_values_list = []
    memory_values_list = []
    lowest_ram_list = []
    inference_cpu_file_name = "inference_cpu_quota_runtime_values.csv"
    inference_memory_file_name = "inference_memory_runtime_values.csv"
    #TODO: automate folder walkthrough
    # subdirs = [x[0] for x in os.walk(dir)]
    for dataset in dataset_name_list:
        ram_big_list = []
        cpu_big_list = []
        print(f"At {dataset}\n")
        for model in model_name_list:
            lowest_model_memory_run = 10000
            cpu_values_list = []
            memory_values_list = []
            tmp_path = os.path.join(dir,dataset,model, inference_cpu_file_name)
            if os.path.exists(tmp_path):
                with open(tmp_path, "r") as csvfile:
                            csvreader = csv.reader(csvfile, delimiter=',')
                            runtime_header = next(csvreader, None)
                            for row in csvreader:
                                cpu_values_list.append(row)

                for runtime_list in cpu_values_list:
                    CPU_qtc= float(runtime_list[1])/100000.0 # qtc=quota_to_core; convert cpu quota to cpu cores for easier interpretation
                    if cpu_limit == None or cpu_limit > CPU_qtc:
                        if model == 'CNN_1d':
                            cpu_big_list.append([CNN_name, float(runtime_list[-1])/10.0, CPU_qtc])                            
                        elif model == 'Alexnet1d':
                            cpu_big_list.append([alexnet_name, float(runtime_list[-1])/10.0, CPU_qtc])
                        elif model == 'Resnet1d':
                            cpu_big_list.append([resnet_name, float(runtime_list[-1])/10.0, CPU_qtc])
                        else:
                            cpu_big_list.append([model, float(runtime_list[-1])/10.0, CPU_qtc])
                    # for function_time in runtime_list[2:]:
                    #     function_time_list.append(float(function_time))
                    #     time_sum+=float(function_time)
                    # time_sum=0.0

            tmp_path = os.path.join(dir,dataset,model, inference_memory_file_name)
            if os.path.exists(tmp_path):
                with open(tmp_path, "r") as csvfile:
                            csvreader = csv.reader(csvfile, delimiter=',')
                            runtime_header = next(csvreader, None)
                            for row in csvreader:
                                memory_values_list.append(row)

                for runtime_list in memory_values_list:
                    ram_big_list.append([model, float(runtime_list[-1]), float(runtime_list[1])])
                    if lowest_model_memory_run > int(runtime_list[1]):
                        lowest_model_memory_run = int(runtime_list[1])
                if lowest_model_memory_run != 10000:
                    if model == 'CNN_1d':
                        lowest_ram_list.append([dataset, CNN_name, lowest_model_memory_run])
                    elif model == 'Alexnet1d':
                        lowest_ram_list.append([dataset, alexnet_name, lowest_model_memory_run])
                    elif model == 'Resnet1d':
                        lowest_ram_list.append([dataset, resnet_name, lowest_model_memory_run])
                    else:
                        lowest_ram_list.append([dataset, model, lowest_model_memory_run])
        cpu_big_df = pd.DataFrame(cpu_big_list, columns=['Model', 'Time(s)', 'CPU Cores'])
        
        cpu_plot = sns.relplot(data=cpu_big_df, x='CPU Cores', y='Time(s)', hue='Model', kind='line', height=7, aspect=5/4)
        save_fig(cpu_plot,f"{dataset}\nCPU Cores vs Inference Time\nFor 1 observation", f'{cwd}/plots/{dataset}_CPU_inf_all.svg')
        plt.figure()
    low_ram_df = pd.DataFrame(lowest_ram_list, columns=['Dataset', 'Model', 'Minimum RAM'])
    ram_bar_plot = sns.barplot(data=low_ram_df, x='Model', y='Minimum RAM', hue='Dataset').set_title(f"Minimum RAM required for inference")
    fig = ram_bar_plot.get_figure()
    fig.savefig(f'{cwd}/plots/RAM_bar_plot_inference.svg')

def cpu_eval_plot(dir, cpu_limit: int = None):
    """Generate combined inference plot of all available models for each individual dataset

    Args:
        dir (str): Path to logs directory. Defaults to cwd.
        cpu_limit (int): Top limit for CPU plotting (e.g 1 core). Defaults to None.
    """
    cwd = os.getcwd()
    cpu_values_list = None
    train_cpu_file_name = "train_cpu_quota_runtime_values.csv"
    train_memory_file_name = "train_memory_runtime_values.csv"
    cpu_big_list = []
    global model_dict
    dataset_folders = f'{cwd}/host_data'
    # dataset_name_list = ['SEU', 'MFPT']
    # model_name_list = ['LeNet1d', 'Alexnet1d', 'Resnet1d', 'MLP', 'CNN_1d', 'BiLSTM1d']
    for dataset_name in dataset_name_list:
        model_folders = os.path.join(dataset_folders,dataset_name)
        for model_name in model_name_list:
            model_dir = os.path.join(model_folders,model_name)
            cpu_values_file = os.path.join(model_dir, train_cpu_file_name)
            if os.path.isfile(cpu_values_file):
                cpu_values_list = []
                with open(cpu_values_file,"r") as csvfile:
                    csvreader = csv.reader(csvfile, delimiter=',')
                    _runtime_header = next(csvreader, None)
                    for row in csvreader:
                        cpu_values_list.append(row)
                for runtime_list in cpu_values_list:
                    CPU_qtc= float(runtime_list[1])/100000.0 # convert cpu quota to cpu cores for easier interpretation
                    if cpu_limit == None or cpu_limit > CPU_qtc:
                        if model_name == 'CNN_1d':
                            cpu_big_list.append([CNN_name, float(runtime_list[-1])/10, CPU_qtc])
                        elif model_name == 'Alexnet1d':
                            cpu_big_list.append([alexnet_name, float(runtime_list[-1])/10, CPU_qtc])
                        elif model_name == 'Resnet1d':
                            cpu_big_list.append([resnet_name, float(runtime_list[-1])/10, CPU_qtc])
                        else:
                            cpu_big_list.append([model_name, float(runtime_list[-1])/10, CPU_qtc])
        cpu_big_df = pd.DataFrame(cpu_big_list, columns=['Model', 'Time(s)', 'CPU Cores'])
        ax1 = sns.relplot(data=cpu_big_df, x='CPU Cores', y='Time(s)', hue='Model', kind='line', height=7, aspect=5/4)
        save_fig(ax1, f"{dataset_name}\nCPU Cores vs Train Time", f"{cwd}/plots/{dataset_name}_CPU_train_all.svg")


def get_subdirs(path):
    #TODO
    pass

def save_fig(figure, title: str = None, path: str = None):
    """Function for saving figures

    Args:
        figure: object to save 
        title (str, optional): Title of the figure. Defaults to None.
        path (str, optional): Path to save figure. Defaults to None.
    """
    with sns.axes_style("white"):
        figure.fig.suptitle(title, y=1.08) # y - used to increase title height 
        figure.savefig(path)
        plt.close()


def remove_outlier(df_in, col_name):
    
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def input_limit():
    """Function that takes user input for the limit of CPU core plot 

    Returns:
        value: Limit that user has inputed
    """
    os.system("stty sane")
    while True:
        try:
            value = float(input("CPU cores plot limit:"))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue

        if value < 0:
            print("Sorry, your response must not be negative.")
            continue
        else:
            break
    return value


if __name__=="__main__":
    # train_flag = False
    # cpu_values_list = None
    # memory_values_list = None
    # if questionary.confirm("Detailed plot execution?").ask():
    #     cpu_values_list, memory_values_list, model_name, dataset_name, train_flag = init()
    # time_sum=0.0

    # RAM_limit_list=[]
    # CPU_limit_list=[]
    # RAM_flag=False
    # CPU_flag=False
    # if memory_values_list != None:
    #     RAM_flag=True
    #     CPU_flag=False
    #     function_time_list=[]
    #     total_time_list=[]
    #     for runtime_list in memory_values_list:
    #         RAM_limit_list.append(float(runtime_list[1]))
    #         for function_time in runtime_list[2:]:
    #             function_time_list.append(float(function_time))
    #             time_sum+=float(function_time)
    #         total_time_list.append(time_sum)
    #         time_sum=0.0
    #     single_plots(function_time_list, total_time_list, RAM_limit_list, model_name, dataset_name, RAM_flag, CPU_flag, train_flag)    
    # else: print("No memory file found!")
    # if cpu_values_list != None:
    #     CPU_flag=True
    #     RAM_flag=False
    #     function_time_list=[]
    #     total_time_list = []
    #     for runtime_list in cpu_values_list:
    #         CPU_qttc= float(runtime_list[1])/100000.0 # convert cpu quota to cpu cores for easier interpretation
    #         CPU_limit_list.append(float(CPU_qttc))
    #         for function_time in runtime_list[2:]:
    #             function_time_list.append(float(function_time))
    #             time_sum+=float(function_time)
    #         total_time_list.append(time_sum)
    #         time_sum=0.0
    #     single_plots(function_time_list, total_time_list, CPU_limit_list, model_name, dataset_name, RAM_flag, CPU_flag, train_flag)
    # else: print("No CPU file found!")
    # if train_flag:
    #     epoch_plot(cpu_runtime_values=cpu_values_list, ram_runtime_values=memory_values_list, model_name=model_name, dataset_name=dataset_name)
    plot_init()
    folder_create(os.path.join(os.getcwd(),'plots'))
    tmp_flag = questionary.confirm("Train overlay plot?").ask()
    cpu_limit = None
    if tmp_flag:
        if questionary.confirm("Choose CPU top limit?").ask():
            cpu_limit = input_limit()
        cwd = os.getcwd()
        logs_dir = os.path.join(cwd,'host_data')
        plot_all_models(logs_dir, cpu_limit)
    cpu_limit = None
    if questionary.confirm("Inference overlay plot?").ask():
        if questionary.confirm("Choose CPU top limit?").ask():
            cpu_limit = input_limit()
        cwd = os.getcwd()
        logs_dir = os.path.join(cwd,'host_data')
        inference_plot(logs_dir, cpu_limit)
    cpu_limit = None
    if questionary.confirm("Train time plot?").ask():
        if questionary.confirm("Choose CPU top limit?").ask():
            cpu_limit = input_limit()
        cwd = os.getcwd()
        logs_dir = os.path.join(cwd,'host_data')
        cpu_eval_plot(logs_dir, cpu_limit)
    print ("Script ran sucessfully!")



# def init():
#     train_flag = False
#     cpu_values_list = None
#     memory_values_list = None
#     dataset_folders = f'{cwd}/host_data'
#     dataset_name = questionary.select("Choose dataset:",choices=os.listdir(dataset_folders)).ask()
#     model_folders = os.path.join(dataset_folders,dataset_name)
#     model_name = questionary.select("Choose model folder:", choices=os.listdir(model_folders)).ask()
#     model_dir = os.path.join(model_folders,model_name)
#     plot_selection = questionary.select("Train or inference:", choices=["train", "inference"]).ask()
#     if plot_selection == "train":
#         train_flag = True
#         cpu_values_file = os.path.join(model_dir, train_cpu_file_name)
#         memory_values_file = os.path.join(model_dir, train_memory_file_name)    
#     else:
#         cpu_values_file = os.path.join(model_dir, inference_cpu_file_name)
#         memory_values_file = os.path.join(model_dir, inference_memory_file_name)
#     if os.path.isfile(cpu_values_file):
#         cpu_values_list = []
#         with open(cpu_values_file,"r") as csvfile:
#             csvreader = csv.reader(csvfile, delimiter=',')
#             runtime_header = next(csvreader, None)
#             for row in csvreader:
#                 cpu_values_list.append(row)
#     if os.path.isfile(memory_values_file):
#         memory_values_list = []
#         with open(memory_values_file, "r") as csvfile:
#             csvreader = csv.reader(csvfile, delimiter=',')
#             runtime_header = next(csvreader, None)
#             for row in csvreader:
#                 memory_values_list.append(row)

#     else: cpu_log_list = None
#     if cpu_values_list != None and memory_values_list != None:
#         return cpu_values_list, memory_values_list, model_name, dataset_name, train_flag
#     elif cpu_values_list != None:
#         return cpu_values_list, None, model_name, dataset_name, train_flag
#     elif memory_values_list != None:
#         return None,memory_values_list, model_name, dataset_name, train_flag
#     else: return None,None, model_name, dataset_name, train_flag
