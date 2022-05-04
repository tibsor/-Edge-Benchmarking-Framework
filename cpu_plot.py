from logging import exception
import matplotlib.pyplot as plt
import numpy as np # for creating memory range list
import os # for getting bash script variables
import csv 



def init():
    cwd = os.getcwd()
    time_list=[]
    CPU_limit_list=[]
    # limit_step=int(os.environ["step"])
    # upper_limit=int(os.environ["upper_limit"])
    # lower_limit=int(os.environ["lower_limit"])
    ##### FOR DEBUG #####
    upper_limit=100000
    lower_limit=1000
    limit_step=1000
    ##### FOR DEBUG #####
    CPU_limit_list=np.arange(upper_limit, lower_limit-1, -limit_step)
    with open(f"{cwd}/host_data/runtime_values.csv","r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        header=next(csvreader, None)  # skip the headers
        for row in csvreader:
            time_list.append(row)
    return CPU_limit_list, time_list

    print (line)

def plot(x_axis: list = None, y_axis: list = None):
    # if x_axis==None:
    #     raise ValueError("X axis list is none!")
    # if y_axis==None:
    #     raise ValueError("Y axis list is none!")   
    plt.plot(x_axis,y_axis, 'ro')
    plt.xlabel('Runtime(seconds)')
    plt.ylabel('CPU cores used')
    plt.title("Time elapsed vs CPU resources")
    plt.show()

if __name__=="__main__":
    CPU_limit_list, time_list = init()
    args_time=init_time=setup_time=eval_time=total_time=0.0

    args_time_list=[]
    init_time_list=[]
    setup_time_list=[]
    eval_time_list=[]
    total_time_list=[]
    for runtime_list in time_list:
        for function_time in runtime_list:
        #print(time)
            total_time+=float(function_time)
        total_time_list.append(float(total_time))
        total_time=0.0
        args_time_list.append(float(runtime_list[0]))
        init_time_list.append(float(runtime_list[1]))
        setup_time_list.append(float(runtime_list[2]))
        eval_time_list.append(float(runtime_list[3]))
        # args_time+=float(runtime_list[0])
        # init_time+=float(runtime_list[1])
        # setup_time+=float(runtime_list[2])
        # eval_time+=float(runtime_list[3])
    plot(total_time_list, CPU_limit_list)
