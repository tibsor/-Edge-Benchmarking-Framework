#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
import models

class inference_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
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

        print(Dataset)

        self.datasets = {}

        self.datasets['val'] = Dataset(args.data_dir,args.normalizetype).data_preprare()

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['val']}
        # Define the model
        self.model = getattr(models, args.model_name)(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes)
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # # Define the optimizer
        # if args.opt == 'sgd':
        #     self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
        #                                 momentum=args.momentum, weight_decay=args.weight_decay)
        # elif args.opt == 'adam':
        #     self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr,
        #                                 weight_decay=args.weight_decay)
        # else:
        #     raise Exception("optimizer not implement")

        # # Define the learning rate decay
        # if args.lr_scheduler == 'step':
        #     steps = [int(step) for step in args.steps.split(',')]
        #     self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        # elif args.lr_scheduler == 'exp':
        #     self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        # elif args.lr_scheduler == 'stepLR':
        #     steps = int(args.steps)
        #     self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        # elif args.lr_scheduler == 'fix':
        #     self.lr_scheduler = None
        # else:
        #     raise Exception("lr schedule not implement")

        # # Load the checkpoint
        # self.start_epoch = 0

        # # Invert the model and define the loss
        # self.model.to(self.device)
        # self.criterion = nn.CrossEntropyLoss()


    def inference(self):
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()



        epoch_start = time.time()
        epoch_acc = 0.8873
        epoch_loss = 0.3475
        phase = 'val'
        self.model.eval()
        











