#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import csv
import os
from datetime import datetime
import time
from utils.logger import setlogger
import logging
from utils.train_utils_ae import train_utils

args = None
time_dict={"parse_args()": None, "create_folder()": None,"train.init()": None, "train.setup()": None, "train.evaluate()": None}
time_list=[]
csv_header=None
vol_path='/inference/volume_data'

def parse_args():
    start = time.time()

    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='Sae1d', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='SEU', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default= "/inference/Mechanical-datasets", help='the directory of the data')
    parser.add_argument('--normalizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1', help='data normalization methods')
    parser.add_argument('--processing_type', type=str, choices=['R_A', 'R_NA', 'O_A'], default='O_A',
                        help='R_A: random split with data augmentation, R_NA: random split without data augmentation, O_A: order split with data augmentation')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=32, help='batchsize of the training process')
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
    parser.add_argument('--middle_epoch', type=int, default=50, help='middle number of epoch')

    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')
    args = parser.parse_args()
    global csv_model_values
    csv_model_values=[now, args.model_name, args.data_name, args.normalizetype, args.processing_type, args.batch_size, args.opt, args.lr, args.steps, args.max_epoch]

    end = time.time()
    print("parse_args() function takes", end-start, "seconds")
    time_dict['parse_args()']=end-start
    time_list.append(end-start)
    return args


def create_folder(model_name: str = None, dataset: str = None):
    start = time.time()
    global memory_limit, cpu_quota, cpu_quota_path, mem_rt_path, csv_header 
    model_folder = os.path.join(vol_path, model_name)
    isdir = os.path.isdir(model_folder)
    if isdir:
        pass
    else:
        os.mkdir(model_folder)
    dataset_folder = os.path.join(model_folder, dataset)
    isdir = os.path.isdir(dataset_folder)
    if isdir:
        pass
    else:
        os.mkdir(dataset_folder)    
    #csv_model_header=["model", "dataset", "normalizetype", "processing_type", "batch_type", "optimizer", "learning_rate", "steps", "max_epoch"]
    
    model_details_path = os.path.join(dataset_folder, "train_run_details.csv")
    if not(os.path.exists(model_details_path)):
        csv_model_header=["timedate", "model", "dataset", "normalizetype", "processing_type", "batch_size", "optimizer", "learning_rate", "steps", "max_epoch"]
        with open(model_details_path,'a+', encoding='UTF8', newline="") as f:
            writer=csv.writer(f)
            writer.writerow(csv_model_header)
            writer.writerow(csv_model_values)
    else:
        with open(model_details_path, "a+", encoding='UTF8', newline="") as f:
            writer=csv.writer(f)
            writer.writerow(csv_model_values)
    
    mem_rt_path=os.path.join(vol_path, dataset_folder, 'train_memory_runtime_values.csv')
    cpu_quota_path=os.path.join(vol_path, dataset_folder, 'train_cpu_quota_runtime_values.csv')
    if "MEM_LIMIT" in os.environ:
        
        memory_limit = int(os.environ["MEM_LIMIT"])
        memory_reserve=memory_limit/2.0
        csv_header=["timedate", "memory_limit","parse_args()", "create_folder()","train.init()", "train.setup()", "train.evaluate()"]
        if not(os.path.exists(mem_rt_path)):
            with open(mem_rt_path,'a+', encoding='UTF8', newline="") as f:
                writer=csv.writer(f)
                writer.writerow(csv_header)
    else: 
        memory_limit=None
        memory_reserve=None

    if "CPU_QUOTA" in os.environ:
        cpu_quota=int(os.environ["CPU_QUOTA"])
        cpu_period=100000
        csv_header=["timedate", "cpu_quota", "parse_args()", "create_folder()","train.init()", "train.setup()", "train.evaluate()"]
        if not(os.path.exists(cpu_quota_path)):
            with open(cpu_quota_path,'a+', encoding='UTF8', newline="") as f:
                writer=csv.writer(f)
                writer.writerow(csv_header)
    else: cpu_quota=None
    end = time.time()
    print("create_folder() function takes", end-start, "seconds")
    time_dict["create_folder()"]=end-start
    time_list.append(end-start)

    return dataset_folder

def question():
    i = 0
    while i < 2:
        answer = input("Save models? (yes or no)\n")
        if any(answer.lower() == f for f in ["yes", 'y', '1', 'ye']):
            print("Yes")
            answer = "yes"
            break
        elif any(answer.lower() == f for f in ['no', 'n', '0']):
            print("No")
            break
        else:
            i += 1
            if i < 2:
                print('Please enter yes or no')
            else:
                print("Nothing done")
    return answer

if __name__ == '__main__':
    now = datetime.now()
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    answer = None
    save_dir = create_folder(model_name=args.model_name, dataset=args.data_name)
    # answer = question()
    # if answer == "yes": 
    #     pass
    # elif answer == "no": 
    #     save_dir = '/dev/null'


    ##### DEBUG: DON'T SAVE MODELS ON HOST MACHINE ######
    #save_dir = '/dev/null'

    ##### DEBUG END: DON'T SAVE MODELS ON HOST MACHINE######
    # set the logger
    if memory_limit:
        setlogger(os.path.join(save_dir,"RAM_training.log"))
    if cpu_quota:
        setlogger(os.path.join(save_dir, 'CPU_training.log'))
    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    values_list=[]
    start = time.time()
    trainer = train_utils(args, save_dir)
    end = time.time()
    print("train.__init__() takes", end-start, "seconds")
    time_dict["train.init()"]=end-start
    time_list.append(end-start)
    start = time.time()    
    trainer.setup()
    end = time.time()
    print("train.setup() takes", end-start, "seconds")
    time_dict["train.setup()"]=end-start
    time_list.append(end-start)
    start = time.time()
    trainer.train()
    end = time.time()
    print("train.evaluate() takes", end-start, "seconds")
    time_dict["train.evaluate()"]=end-start
    time_list.append(end-start)
    print(time_dict,"\n")
    if csv_header != None:
        values_list.append(now)
    else: raise ValueError("CSV HEADER IS NONE!")
    if memory_limit!=None:
        values_list.append(memory_limit)
        values_list.append(time_dict['parse_args()'])
        values_list.append(time_dict['create_folder()'])
        values_list.append(time_dict['train.init()'])
        values_list.append(time_dict['train.setup()'])
        values_list.append(time_dict['train.evaluate()'])
        with open(mem_rt_path,'a+', encoding='UTF8', newline="") as f:
            writer=csv.writer(f, delimiter=',')
            writer.writerow(values_list)
    if cpu_quota!=None:
        values_list.append(cpu_quota)
        values_list.append(time_dict['parse_args()'])
        values_list.append(time_dict['create_folder()'])
        values_list.append(time_dict['train.init()'])
        values_list.append(time_dict['train.setup()'])
        values_list.append(time_dict['train.evaluate()'])
        with open(cpu_quota_path,'a+', encoding='UTF8', newline="") as f:
            writer=csv.writer(f, delimiter=',')
            writer.writerow(values_list)






