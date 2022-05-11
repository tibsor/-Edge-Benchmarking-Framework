from logging import exception
import matplotlib.pyplot as plt
import numpy as np # for creating memory range list
import os # for getting bash script variables
import csv 



def init():
    cwd = os.getcwd()
    values_list=[]
    ##### FOR DEBUG #####
    #TODO: automate path with menu
    global model_dict
    model_dict = {"model": None,"dataset": None, "normalizetype": None,"processing_type": None}

    #with open(f"{cwd}/host_data/runtime_values.csv","r") as csvfile:
    with open("/home/tibs/edge-benchmarking-framework/host_data/MLP/SEU/cpu_quota_runtime_values.csv","r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        model_header=next(csvreader, None)  # [model,dataset,normalizetype,processing_type]
        model_values=next(csvreader, None)
        model_dict['model']=model_values[0]
        model_dict['dataset']=model_values[1]
        model_dict['normalizetype']=model_values[2]
        model_dict['processing_type']=model_values[3]
        runtime_header = next(csvreader, None)
        for row in csvreader:
            values_list.append(row)
    return values_list

    print (line)

def plot(x_axis: list = None, y_axis: list = None):
    # if x_axis==None:
    #     raise ValueError("X axis list is none!")
    # if y_axis==None:
    #     raise ValueError("Y axis list is none!") 
    main_plot = plt.figure(2)  
    plt.plot(x_axis,y_axis, 'ro')
    plt.xlabel('Time (s)')
    plt.ylabel('CPU quota (us)')
    plt.title("Total time elapsed vs CPU quota")
    plt.show()



def function_time_plots(x_axis: list = None, x1_axis: list = None, y_axis: list = None):
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
    fig.suptitle('Function time vs CPU Quota')

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
    plt.show()


if __name__=="__main__":
    values_list = init()
    total_time=0.0
    function_time_list=[]
    total_time_list=[]
    CPU_limit_list=[]
    for runtime_list in values_list:
        CPU_limit_list.append(float(runtime_list[1]))
        for function_time in runtime_list[2:]:
            function_time_list.append(float(function_time))
            total_time+=float(function_time)
        total_time_list.append(float(total_time))
        total_time=0.0
    
    function_time_plots(function_time_list, total_time_list, CPU_limit_list)
    #plot(total_time_list, CPU_limit_list)
