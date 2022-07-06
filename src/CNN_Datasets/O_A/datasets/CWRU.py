# Data science libraries
import os
import scipy.io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Pytorch
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss

# Others
from pathlib import Path

#from helper import get_df_all
# from train_helper import get_dataloader, fit, validate 
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *

# Pytorch
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

working_dir = Path('/benchmark/')
DATA_PATH = Path("/benchmark/CWRU/")
# save_model_path = working_dir / 'Model'
DE_path = DATA_PATH / '12k_FE'

bs = 64

# Functions for training
def get_dataloader(train_ds, valid_ds, bs):
    '''
        Get dataloaders of the training and validation set.
        Parameter:
            train_ds: Dataset
                Training set
            valid_ds: Dataset
                Validation set
            bs: Int
                Batch size
        
        Return:
            (train_dl, valid_dl): Tuple of DataLoader
                Dataloaders of training and validation set.
    '''
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def matfile_to_dic(folder_path):
    '''
    Read all the matlab files of the CWRU Bearing Dataset and return a 
    dictionary. The key of each item is the filename and the value is the data 
    of one matlab file, which also has key value pairs.
    
    Parameter:
        folder_path: 
            Path (Path object) of the folder which contains the matlab files.
    Return:
        output_dic: 
            Dictionary which contains data of all files in the folder_path.
    '''
    output_dic = {}
    for _, filepath in enumerate(folder_path.glob('*.mat')):
        # strip the folder path and get the filename only.
        key_name = str(filepath).split('\\')[-1]
        output_dic[key_name] = scipy.io.loadmat(filepath)
    return output_dic


def remove_dic_items(dic):
    '''
    Remove redundant data in the dictionary returned by matfile_to_dic inplace.
    '''
    # For each file in the dictionary, delete the redundant key-value pairs
    for _, values in dic.items():
        del values['__header__']
        del values['__version__']    
        del values['__globals__']


def rename_keys(dic):
    '''
    Rename some keys so that they can be loaded into a 
    DataFrame with consistent column names
    '''
    # For each file in the dictionary
    for _,v1 in dic.items():
        # For each key-value pair, rename the following keys 
        for k2,_ in list(v1.items()):
            if 'DE_time' in k2:
                v1['DE_time'] = v1.pop(k2)
            elif 'BA_time' in k2:
                v1['BA_time'] = v1.pop(k2)
            elif 'FE_time' in k2:
                v1['FE_time'] = v1.pop(k2)
            elif 'RPM' in k2:
                v1['RPM'] = v1.pop(k2)


def label(filename):
    '''
    Function to create label for each signal based on the filename. Apply this
    to the "filename" column of the DataFrame.
    Usage:
        df['label'] = df['filename'].apply(label)
    '''
    if 'B' in filename:
        return 'B'
    elif 'IR' in filename:
        return 'IR'
    elif 'OR' in filename:
        return 'OR'
    elif 'Normal' in filename:
        return 'N'


def matfile_to_df(folder_path):
    '''
    Read all the matlab files in the folder, preprocess, and return a DataFrame
    
    Parameter:
        folder_path: 
            Path (Path object) of the folder which contains the matlab files.
    Return:
        DataFrame with preprocessed data
    '''
    dic = matfile_to_dic(folder_path)
    remove_dic_items(dic)
    rename_keys(dic)
    df = pd.DataFrame.from_dict(dic).T
    df = df.reset_index().rename(mapper={'index':'filename'},axis=1)
    df['label'] = df['filename'].apply(label)
    return df.drop(['BA_time','FE_time', 'RPM', 'ans'], axis=1, errors='ignore')


def divide_signal(df, segment_length):
    '''
    This function divide the signal into segments, each with a specific number 
    of points as defined by segment_length. Each segment will be added as an 
    example (a row) in the returned DataFrame. Thus it increases the number of 
    training examples. The remaining points which are less than segment_length 
    are discarded.
    
    Parameter:
        df: 
            DataFrame returned by matfile_to_df()
        segment_length: 
            Number of points per segment.
    Return:
        DataFrame with segmented signals and their corresponding filename and 
        label
    '''
    dic = {}
    idx = 0
    for i in range(df.shape[0]):
        n_sample_points = len(df.iloc[i,1])
        n_segments = n_sample_points // segment_length
        for segment in range(n_segments):
            dic[idx] = {
                'signal': df.iloc[i,1][segment_length * segment:segment_length * (segment+1)], 
                'label': df.iloc[i,2],
                'filename' : df.iloc[i,0]
            }
            idx += 1
    df_tmp = pd.DataFrame.from_dict(dic,orient='index')
    df_output = pd.concat(
        [df_tmp[['label', 'filename']], 
         pd.DataFrame(np.hstack(df_tmp["signal"].values).T)
        ], 
        axis=1 )
    return df_output


def normalize_signal(df):
    '''
    Normalize the signals in the DataFrame returned by matfile_to_df() by subtracting
    the mean and dividing by the standard deviation.
    '''
    mean = df['DE_time'].apply(np.mean)
    std = df['DE_time'].apply(np.std)
    df['DE_time'] = (df['DE_time'] - mean) / std


def get_df_all(data_path, segment_length=512, normalize=False):
    '''
    Load, preprocess and return a DataFrame which contains all signals data and
    labels and is ready to be used for model training.
    
    Parameter:
        normal_path: 
            Path of the folder which contains matlab files of normal bearings
        DE_path: 
            Path of the folder which contains matlab files of DE faulty bearings
        segment_length: 
            Number of points per segment. See divide_signal() function
        normalize: 
            Boolean to perform normalization to the signal data
    Return:
        df_all: 
            DataFrame which is ready to be used for model training.
    '''
    df = matfile_to_df(data_path)

    if normalize:
        normalize_signal(df)
    df_processed = divide_signal(df, segment_length)

    map_label = {'N':0, 'B':1, 'IR':2, 'OR':3}
    df_processed['label'] = df_processed['label'].map(map_label)
    return df_processed



def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
            Reshape(),
            Normalize(normlize_type),
            RandomAddGaussian(),
            RandomScale(),
            RandomStretch(),
            RandomCrop(),
            Retype()

        ]),
        'val': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]


class CWRU(object):
    num_classes = 10
    inputchannel = 1

    def __init__(self, data_dir,normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_preprare(self, test=False):

        # list_data = get_files(self.data_dir, test)
        df_all = get_df_all(DE_path, segment_length=500, normalize=True)
        features = df_all.columns[2:]
        target = 'label'

        ## Split the data into train and validation set
        X_train, X_valid, y_train, y_valid = train_test_split(df_all[features], 
                                                            df_all[target], 
                                                            test_size=0.20, random_state=0, shuffle=True
                                                            )
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        X_valid = torch.tensor(X_valid.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        y_valid = torch.tensor(y_valid.values, dtype=torch.long)

        train_ds = TensorDataset(X_train, y_train)
        valid_ds = TensorDataset(X_valid, y_valid)
        train_dl, valid_dl = get_dataloader(train_ds, valid_ds, bs)
        ## Create DataLoader of train and validation set
        
        if test:
            test_dataset = dataset(list_data=df_all, test=True, transform=None)
            return test_dataset
        else:
            train_dataset = train_dl.dataset
            val_dataset = valid_dl.dataset
            return train_dataset, val_dataset
