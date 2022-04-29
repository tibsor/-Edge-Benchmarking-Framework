from logging import exception
import matplotlib.pyplot as plt
import numpy as np # for creating memory range list
import os # for getting bash script variables
import csv 



def init():
    cwd = os.getcwd()
    time_list=[]
    RAM_limit_list=[]
    limit_step=int(os.environ["step"])
    upper_limit=int(os.environ["upper_limit"])
    lower_limit=int(os.environ["lower_limit"])
    ##### FOR DEBUG #####
    # upper_limit=164
    # lower_limit=80
    # limit_step=2
    ##### FOR DEBUG #####
    RAM_limit_list=np.arange(upper_limit, lower_limit-1, -limit_step)
    with open(f"{cwd}/host_data/runtime_values.csv","r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        header=next(csvreader, None)  # skip the headers
        for row in csvreader:
            time_list.append(row)
    return RAM_limit_list, time_list

    print (line)

def main(x_axis: list = None, y_axis: list = None):
    # if x_axis==None:
    #     raise ValueError("X axis list is none!")
    # if y_axis==None:
    #     raise ValueError("Y axis list is none!")
    plt.plot(x_axis,y_axis, 'ro')
    plt.xlabel('Runtime(seconds)')
    plt.ylabel('RAM (MBs)')
    plt.show()

if __name__=="__main__":
    RAM_limit_list, time_list = init()
    time_sum=0.0
    total_time_list=[]
    for runtime_list in time_list:
        for function_time in runtime_list:
        #print(time)
            time_sum+=float(function_time)
        total_time_list.append(time_sum)
        time_sum=0.0

    main(total_time_list, RAM_limit_list)