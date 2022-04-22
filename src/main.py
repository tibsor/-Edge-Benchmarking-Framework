# importing libraries
import sys
import os
import warnings
import argparse
from utils.logger import setlogger
import logging
import models
import torch
from datetime import datetime
import random
args = None

import psutil
# Getting % usage of virtual_memory
print('System RAM % used:', psutil.virtual_memory()[2])


def parse_args():

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
    return args

class inference(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir
    def setup(self):
        """
        Initialize the dataset, model
        :return:
        """
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))
        
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
        print('System RAM % used in setup:', psutil.virtual_memory()[2])


    def evaluate(self):
        
        for phase in ['val']:
            valid_loss = 0.0
            self.model.eval()
            now = datetime.now()
            random.seed(2022)
            randomlist = random.sample(range(0, 419), 10)
            with open("values.txt","w") as f:
                f.write(f"{str(now)}\n")
            # Getting % usage of virtual_memory ( 3rd field)
            print('System RAM % used in inference:', psutil.virtual_memory()[2])
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
                            with open("values.txt","a") as f:
                                f.write(f"{batch_id}; Correct: {correct}; loss: {loss_temp}\n")
                            print(f"Correct: {correct}, Loss Temp: {loss_temp}\n")
                            break
            # Getting % usage of virtual_memory
            print('System RAM % used in inference:', psutil.virtual_memory()[2])

if __name__ == '__main__':
    args = parse_args()

    save_dir = os.path.join(args.checkpoint_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'inference.log'))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        eval_output = inference(args, save_dir)
        eval_output.setup()
        eval_output.evaluate()
        # while True:
        #     pass
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)