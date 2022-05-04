# importing libraries
import sys
import os
import time
import warnings
import argparse
from utils.logger import setlogger
import models
import csv
import torch
from datetime import datetime
import random
from memory_profiler import profile
args = None
time_dict={"args_time":None, "init_time":None, "setup_time":None, "eval_time":None }
time_list=[]
import psutil
vol_path='/inference/volume_data'
mem_rt_path=os.path.join(vol_path, 'memory_runtime_values.csv')
mem_profiler_path=os.path.join(vol_path,'memory_profiler.log')
cpu_quota_path=os.path.join(vol_path,'cpu_quota_runtime_values.csv')
csv_header=None
if "MEM_LIMIT" in os.environ:
    memory_limit=int(os.environ["MEM_LIMIT"])
    memory_reserve=memory_limit/2.0
    csv_header=["timedate", "memory_limit", "args_time", "init_time", "setup_time", "eval_time"]
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
    csv_header=["timedate", "cpu_quota", "args_time", "init_time", "setup_time", "eval_time"]
else: cpu_quota=None
# Getting % usage of virtual_memory
# print('System RAM % used:', psutil.virtual_memory()[2])
mem_log_file=open(mem_profiler_path,'a+')




@profile(stream=mem_log_file)
def parse_args():
    start = time.time()
    
    parser = argparse.ArgumentParser(description='Inference')
    # basic parameters
    parser.add_argument('--model_name', type=str, default='MLP', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='SEU', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default= "/inference/Mechanical-datasets", help='the directory of the data')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1', help='data normalization methods')
    parser.add_argument('--processing_type', type=str, choices=['R_A', 'R_NA', 'O_A'], default='O_A',
                        help='R_A: random split with data augmentation, R_NA: random split without data augmentation, O_A: order split with data augmentation')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    args = parser.parse_args()
    end = time.time()
    print("parse_args() function takes", end-start, "seconds")
    time_dict["args_time"]=end-start
    time_list.append(end-start)
    return args

class inference(object):
    @profile(stream=mem_log_file)
    def __init__(self, args, save_dir):
        start = time.time()
        self.args = args
        self.save_dir = save_dir
        end = time.time()
        print("class __init__() function takes", end-start, "seconds")
        time_dict["init_time"]=end-start
        time_list.append(end-start)


    @profile(stream=mem_log_file)
    def setup(self):
        """
        Initialize the dataset, model
        :return:
        """
        start = time.time()
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available, using CPU")
            self.device = torch.device("cpu")
            self.device_count = 1
        
        # Load the datasets
        if args.processing_type == 'O_A':
            from CNN_Datasets.O_A import datasets
            Dataset = getattr(datasets, args.data_name)
        elif args.processing_type == 'R_A':
            from CNN_Datasets.R_A import datasets
            Dataset = getattr(datasets, args.data_name)
        elif args.processing_type == 'R_NA':
            from CNN_Datasets.R_NA import datasets
            Dataset = getattr(datasets, args.data_name)
        else:
            raise Exception("processing type not implement")    
        
        val_dataset_path = '/inference/val_dataset.h5'
        val_dataset = torch.load(val_dataset_path) 
        self.val_dataloader = {'val': torch.utils.data.DataLoader(val_dataset)}

        self.model = getattr(models, args.model_name)(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes)
        model_checkpoint = torch.load('/inference/81-0.6310-best_model.pth', map_location=torch.device('cpu'))
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k,v in model_checkpoint.items():
            name = k.replace("module.", "") #remove module string
            new_state_dict[name] = v
        self.model.load_state_dict(model_checkpoint, strict=False)
        self.criterion = torch.nn.CrossEntropyLoss()
        # Getting % usage of virtual_memory ( 3rd field)
        # print('System RAM % used in setup:', psutil.virtual_memory()[2])
        end = time.time()
        print("class setup() function takes", end-start, "seconds")
        time_dict["setup_time"]=end-start
        time_list.append(end-start)
        

    @profile(stream=mem_log_file)
    def evaluate(self):
        start = time.time()
        for phase in ['val']:
            valid_loss = 0.0
            self.model.eval()
            now = datetime.now()
            random.seed(2022)
            randomlist = random.sample(range(0, 419), 10)
            # with open("/inference/volume_data/values.txt","a+") as f:
            #     f.write(f"{str(now)}\n")
            # Getting % usage of virtual_memory ( 3rd field)
            # print('System RAM % used in inference:', psutil.virtual_memory()[2])
            with torch.set_grad_enabled(phase == 'train'):
             
     
                for batch_id, (data, labels) in enumerate(self.val_dataloader[phase]):
                                    # forward
                    for val in randomlist:
                        # Getting % usage of virtual_memory ( 3rd field)
                        #print(f"RAM memory % used in for {batch_id}:{psutil.virtual_memory()[2]}")
                        if val == batch_id:
                            logits = self.model(data)
                            loss = self.criterion(logits, labels)
                            pred = logits.argmax(dim=1)
                            correct = torch.eq(pred, labels).float().sum().item()
                            loss_temp = loss.item() * data.size(0)
                            # with open("/inference/volume_data/values.txt","a") as f:
                            #     f.write(f"{batch_id}; Correct: {correct}; loss: {loss_temp}\n")
                            #print(f"Correct: {correct}, Loss Temp: {loss_temp}\n")
                            break
            # Getting % usage of virtual_memory
            # print('System RAM % used in inference:', psutil.virtual_memory()[2])
        end = time.time()
        print("class inference() function takes", end-start, "seconds")
        time_dict["eval_time"]=end-start
        time_list.append(end-start)

if __name__ == '__main__':
    args = parse_args()
    values_list=[]
    try:
        eval_output = inference(args, vol_path)
        eval_output.setup()
        eval_output.evaluate()
        mem_log_file.close()
        print(time_dict,"\n")
        if csv_header != None:
            values_list.append(datetime.now())
        else: raise("FATAL ERROR!")
        if memory_limit!=None:
            values_list.append(memory_limit)
        if cpu_quota!=None:
            values_list.append(cpu_quota)
        values_list.append(time_dict['args_time'])
        values_list.append(time_dict['init_time'])
        values_list.append(time_dict['setup_time'])
        values_list.append(time_dict['eval_time'])
        with open(mem_rt_path,'a+', encoding='UTF8', newline="") as f:
            writer=csv.writer(f, delimiter=',')
            writer.writerow(values_list)

        # print("Reached while loop")
        # while True:
        #     pass
    except KeyboardInterrupt:
        sys.stderr.write("Interrupt detected...")
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)