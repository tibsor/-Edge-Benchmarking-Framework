# importing libraries
import shutil
import sys
import os
import time
import argparse
from sklearn.utils import shuffle
from utils.logger import setlogger
import csv
import logging
from datetime import datetime
args = None
time_dict={"parse_args()": None, "create_folder()": None,"inference.init()": None, "inference.setup()": None, "inference.evaluate()": None}
time_list=[]
csv_header=None
vol_path='/benchmark/volume_data'
dataset_folder_name = {"SEU":"Mechanical-datasets", "MFPT":"MFPT_Fault_Data_Sets"}

#mem_profiler_path=os.path.join(vol_path,'memory_profiler.log')
global now


global run_type


def parse_args():

    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='MLP', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='SEU', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default= "/benchmark/", help='the directory of the data')
    parser.add_argument('--normalizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1', help='data normalization methods')
    parser.add_argument('--processing_type', type=str, choices=['R_A', 'R_NA', 'O_A'], default='O_A',
                        help='R_A: random split with data augmentation, R_NA: random split without data augmentation, O_A: order split with data augmentation')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='fix', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='9', help='the learning rate decay for step and stepLR')
    parser.add_argument('--steps1', type=str, default='50,80',
                        help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--middle_epoch', type=int, default=10, help='middle number of epoch')

    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=10, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')
    args = parser.parse_args()
    if args.data_dir == '/benchmark/':
        args.data_dir = os.path.join("/benchmark/", dataset_folder_name[args.data_name])
    global csv_model_values
    csv_model_values=[now, args.model_name, args.data_name, args.normalizetype, args.processing_type, args.batch_size, args.opt, args.lr, args.steps, args.max_epoch]


    return args

def folder_check(folder_path: str = None):
    isdir = os.path.isdir(folder_path)
    if isdir:
        pass
    else:
        os.mkdir(folder_path)
    
def create_folder(model_name: str = None, dataset: str = None):
    global memory_limit, cpu_quota, cpu_quota_path, mem_rt_path, csv_header, model_folder

    folder_check(vol_path)
    dataset_folder = os.path.join(vol_path, dataset)
    folder_check(dataset_folder)    
    model_folder = os.path.join(dataset_folder, model_name)
    folder_check(model_folder)
    now = datetime.now()
    date_folder = os.path.join(model_folder,f'{now.year}_{now.month}_{now.day}')
    folder_check(date_folder)
    mem_rt_path=os.path.join(model_folder, 'inference_memory_runtime_values.csv')
    cpu_quota_path=os.path.join(model_folder, 'inference_cpu_quota_runtime_values.csv')
    if "MEM_LIMIT" in os.environ:
        RAM_log_folder = os.path.join(date_folder,"RAM_log")
        folder_check(RAM_log_folder)
        memory_limit = int(os.environ["MEM_LIMIT"])
        csv_header=["timedate", "memory_limit","parse_args()", "create_folder()","inference.init()", "inference.setup()", "inference.evaluate()"]
        if not(os.path.exists(mem_rt_path)):
            with open(mem_rt_path,'a+', encoding='UTF8', newline="") as f:
                writer=csv.writer(f)
                writer.writerow(csv_header)
    else: 
        memory_limit=None

    if "CPU_QUOTA" in os.environ:
        CPU_log_folder = os.path.join(date_folder,"CPU_log")
        folder_check(CPU_log_folder)
        cpu_quota=int(os.environ["CPU_QUOTA"])
        cpu_period=100000
        csv_header=["timedate", "cpu_quota", "parse_args()", "create_folder()","inference.init()", "inference.setup()", "inference.evaluate()"]
        if not(os.path.exists(cpu_quota_path)):
            with open(cpu_quota_path,'a+', encoding='UTF8', newline="") as f:
                writer=csv.writer(f)
                writer.writerow(csv_header)
    else: cpu_quota=None
    
    return model_folder


if __name__ == '__main__':
    
    now = datetime.now()
    start = time.time()
    args = parse_args()
    end = time.time()
    print("parse_args() function takes", end-start, "seconds")
    time_dict['parse_args()']=end-start
    time_list.append(end-start)
    CNN_datasets = []
    CNN_datasets.append("MLP")
    CNN_datasets.append("CNN_1d")
    CNN_datasets.append("CNN_2d")
    CNN_datasets.append("Resnet1d")
    CNN_datasets.append("Alexnet1d")
    CNN_datasets.append("BiLSTM1d")
    CNN_datasets.append("LeNet1d")


    AE_dataset_flag = True
    from utils.inference_utils import inference_utils


    start = time.time()
    save_dir = create_folder(model_name=args.model_name, dataset=args.data_name)
    end = time.time()
    print("create_folder() takes", end-start, "seconds")
    time_dict["create_folder()"]=end-start
    # set the logger
    cwd = os.getcwd()
    if memory_limit:
        setlogger(os.path.join(cwd,"RAM_inference.log"))
    if cpu_quota:
        setlogger(os.path.join(cwd, 'CPU_inference.log'))
    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    values_list=[]
    start = time.time()
    trainer = inference_utils(args, save_dir)
    end = time.time()
    print("inference.__init__() takes", end-start, "seconds")
    time_dict["inference.init()"]=end-start
    time_list.append(end-start)
    start = time.time()    
    trainer.setup()
    end = time.time()
    print("inference.setup() takes", end-start, "seconds")
    time_dict["inference.setup()"]=end-start
    time_list.append(end-start)
    start = time.time()
    trainer.evaluate()
    end = time.time()
    print("inference.evaluate() takes", end-start, "seconds")
    time_dict["inference.evaluate()"]=end-start
    time_list.append(end-start)
    print(time_dict,"\n")
    if csv_header != None:
        values_list.append(now)
    else: raise ValueError("CSV HEADER IS NONE!")
    if memory_limit!=None:
        values_list.append(memory_limit)
        values_list.append(time_dict['parse_args()'])
        values_list.append(time_dict['create_folder()'])
        values_list.append(time_dict['inference.init()'])
        values_list.append(time_dict['inference.setup()'])
        values_list.append(time_dict['inference.evaluate()'])
        with open(mem_rt_path,'a+', encoding='UTF8', newline="") as f:
            writer=csv.writer(f, delimiter=',')
            writer.writerow(values_list)
    if cpu_quota!=None:
        values_list.append(cpu_quota)
        values_list.append(time_dict['parse_args()'])
        values_list.append(time_dict['create_folder()'])
        values_list.append(time_dict['inference.init()'])
        values_list.append(time_dict['inference.setup()'])
        values_list.append(time_dict['inference.evaluate()'])
        with open(cpu_quota_path,'a+', encoding='UTF8', newline="") as f:
            writer=csv.writer(f, delimiter=',')
            writer.writerow(values_list)
    #date_folder = os.path.join(model_folder,f'{now.year}_{now.month}_{now.day}')
    if memory_limit:
        shutil.copy(os.path.join(cwd,"RAM_inference.log"),f'{vol_path}/{args.data_name}/{args.model_name}/{now.year}_{now.month}_{now.day}/RAM_log/{now.hour}_{now.minute}_RAM_inference.log')
    if cpu_quota:
        shutil.copy(os.path.join(cwd,"CPU_inference.log"),f'{vol_path}/{args.data_name}/{args.model_name}/{now.year}_{now.month}_{now.day}/CPU_log/{now.hour}_{now.minute}_CPU_inference.log')



