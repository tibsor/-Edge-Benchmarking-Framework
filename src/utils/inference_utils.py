import itertools
import logging
import os
import random
import torch
import warnings
import numpy as np 
import models
vol_path = '/benchmark/volume_data'
class inference_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the dataset, model
        :return:
        """
        #start = time.time()
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
        
        self.datasets = {}
        if os.path.exists(f'{vol_path}/{args.data_name}/{args.data_name}_dataset.h5'):
            self.dataloaders = torch.load(f'{vol_path}/{args.data_name}/{args.data_name}_dataset.h5') # mmap mode helps keep dataset off RAM
            #self.datasets['val'] = self.dataloaders['val']

        else:
            self.datasets['train'], self.datasets['val'] = Dataset(args.data_dir,args.normalizetype).data_preprare()

            self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False)) for x in ['train', 'val']}

            torch.save(self.dataloaders,f'{vol_path}/{args.data_name}/{args.data_name}_dataset.h5')

        # val_dataset_path = '/inference/val_dataset.h5'
        # val_dataset = torch.load(val_dataset_path) 
        # global batch_size
        # batch_size = 1
        # self.val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size, shuffle=False)

        self.model = getattr(models, args.model_name)
        # Define the model
        if args.model_name == 'CNN_1d' or args.model_name == 'CNN_2d':
            self.model = self.model.CNN(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes)
        elif args.model_name == 'Alexnet1d':
            self.model = self.model.AlexNet(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes)
        elif args.model_name == 'Resnet1d':
            self.model = self.model.resnet18(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes)
        elif args.model_name == 'BiLSTM1d':
            self.model = getattr(models, args.model_name)(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes)
        elif args.model_name == 'LeNet1d':
            self.model = self.model.LeNet(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes)
        else:
            self.model = getattr(models, args.model_name)(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes)
        
        
        tmp_path = os.path.join(vol_path, self.args.data_name, self.args.model_name)
        model_to_load = None
        for file in os.listdir(tmp_path):
            if file.endswith(".pth"):
                model_to_load = file
        if model_to_load == None:
            raise ValueError("No model found! Please train one before")
        model_checkpoint = torch.load(os.path.join(tmp_path,model_to_load), map_location=torch.device('cpu'))
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k,v in model_checkpoint.items():
            name = k.replace("module.", "") #remove module string
            new_state_dict[name] = v
        self.model.load_state_dict(model_checkpoint, strict=False)
        self.criterion = torch.nn.CrossEntropyLoss()

    def evaluate(self, no_obs: int = 1):
        #start = time.time()
        for phase in ['val']:
            valid_loss = 0.0
            self.model.eval()
            #TODO: Completely random, no seed
            # random.seed(2022)
            randomlist = random.sample(range(0, len(self.dataloaders['val'].dataset.labels)-1), no_obs)
            with torch.set_grad_enabled(phase == 'train'):

                for val in randomlist:

                    sample_at = val
                    k = int(np.floor(sample_at/self.args.batch_size))

                    data,labels = next(itertools.islice(self.dataloaders['val'], k, None))

                    logits = self.model(data)
                    loss = self.criterion(logits, labels)
                    pred = logits.argmax(dim=1)
                    correct = torch.eq(pred, labels).float().sum().item()
                    loss_temp = loss.item() * data.size(0)
                    logging.info(f"Correct: {correct}, Loss Temp: {loss_temp}\n")
