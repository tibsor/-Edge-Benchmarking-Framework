from logging import exception
import matplotlib.pyplot as plt
import numpy as np # for creating memory range list
import os # for getting bash script variables
import csv 



def init():
    cwd = os.getcwd()
    values_list=[]
    #RAM_limit_list=[]
    # limit_step=int(os.environ["step"])
    # upper_limit=int(os.environ["upper_limit"])
    # lower_limit=int(os.environ["lower_limit"])
    ##### FOR DEBUG #####
    # upper_limit=256
    # lower_limit=254
    # limit_step=2
    ##### FOR DEBUG #####
    #RAM_limit_list=np.arange(upper_limit, lower_limit-1, -limit_step)
    with open(f"{cwd}/host_data/memory_runtime_values.csv","r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        header=next(csvreader, None)  # skip the headers
        for row in csvreader:
            values_list.append(row)
    return values_list

    print (line)

def main(x_axis: list = None, y_axis: list = None):
    # if x_axis==None:
    #     raise ValueError("X axis list is none!")
    # if y_axis==None:
    #     raise ValueError("Y axis list is none!")
    main_plot=plt.figure(2)
    plt.plot(x_axis,y_axis, 'ro')
    plt.xlabel('Runtime(seconds)')
    plt.ylabel('RAM (MBs)')
    plt.show()

def function_time_plots(x_axis: list = None, y_axis: list = None):
    # if x_axis==None:
    #     raise ValueError("X axis list is none!")
    # if y_axis==None:
    #     raise ValueError("Y axis list is none!")
    args_time_list=[]
    init_time_list=[]
    setup_time_list=[]
    eval_time_list=[]
    #plt.ion()
    for index,i in enumerate(x_axis):
        if index%4==0:
            args_time_list.append(i)
        elif index%4==1:
            init_time_list.append(i)
        elif index%4==2:
            setup_time_list.append(i)
        elif index%4==3:
            eval_time_list.append(i)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(args_time_list, y_axis, marker='v', linestyle='')
    axs[0, 0].set_title('args time')
    axs[0, 1].plot(init_time_list, y_axis, 'tab:orange', marker='.', linestyle='')
    axs[0, 1].set_title('init time')
    axs[1, 0].plot(setup_time_list, y_axis, 'tab:green',marker='o', linestyle='')
    axs[1, 0].set_title('setup time')
    axs[1, 1].plot(eval_time_list, y_axis, 'tab:red', marker='^', linestyle='')
    axs[1, 1].set_title('eval time')
    fig.tight_layout()
    for ax in axs.flat:
        ax.set(xlabel='Time (s)', ylabel='RAM (MBs)')
    plt.savefig("fct_rt.png")
    

if __name__=="__main__":
    values_list = init()
    time_sum=0.0
    function_time_list=[]
    total_time_list=[]
    RAM_limit_list=[]
    for runtime_list in values_list:
        RAM_limit_list.append(float(runtime_list[1]))
        for function_time in runtime_list[2:]:
        #print(time)
            function_time_list.append(float(function_time))
            time_sum+=float(function_time)
        total_time_list.append(time_sum)
        time_sum=0.0
    
    function_time_plots(function_time_list,RAM_limit_list)
    main(total_time_list, RAM_limit_list)