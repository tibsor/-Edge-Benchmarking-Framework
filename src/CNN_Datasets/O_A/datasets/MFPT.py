import os
import re
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm

signal_size=1024

#label
label1 = [1,2,3,4,5,6,7]
label2 = [ 8,9,10,11,12,13,14]   #The failure data is labeled 1-14

#generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    m = os.listdir(root)
    #datasetname = os.listdir(os.path.join("/tmp", root))  # '1 - Three Baseline Conditions'
    # '2 - Three Outer Race Fault Conditions'
    # '3 - Seven More Outer Race Fault Conditions'
    # '4 - Seven Inner Race Fault Conditions'
    # '5 - Analyses',
    # '6 - Real World Examples
    # Generate a list of data
    pattern_1 = 'Three Baseline Conditions'
    pattern_2 = 'Seven More Outer Race Fault Conditions'
    pattern_3 = 'Seven Inner Race Fault Conditions'
    pattern_list = [pattern_1, pattern_2, pattern_3]
    for pattern in pattern_list:
        for text in m:
            if re.search(pattern, text):
                if pattern == pattern_1:
                    data_root1 = os.path.join('/tmp',root,text)#Path of Three Baseline Conditions
                elif pattern == pattern_2:
                    data_root2 = os.path.join('/tmp',root,text)
                elif pattern == pattern_3:
                    data_root3 = os.path.join('/tmp',root,text)
                break
    dataset1 = [ f for f in os.listdir(data_root1) if f.endswith(".mat") ]  # 'Three Baseline Conditions'
    dataset2 = [ f for f in os.listdir(data_root2) if f.endswith(".mat") ]  # 'Seven More Outer Race Fault Conditions'
    dataset3 = [ f for f in os.listdir(data_root3) if f.endswith(".mat") ]  # 'Seven Inner Race Fault Conditions'

    path1=os.path.join('/tmp',data_root1,dataset1[0])
    data, lab = data_load(path1,label=0)  #The label for normal data is 0

    for i in tqdm(range(len(dataset2))):
        path2=os.path.join('/tmp',data_root2,dataset2[i])
        data1, lab1 = data_load(path2,label=label1[i])
        data += data1
        lab += lab1

    for j in tqdm(range(len(dataset3))):
        path3=os.path.join('/tmp',data_root3,dataset3[j])

        data2, lab2  = data_load(path3,label=label2[j])
        data += data2
        lab += lab2

    return [data,lab]

def data_load(filename,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    if label==0:
        fl = (loadmat(filename)["bearing"][0][0][1])     #Take out the data
    else:
        fl = (loadmat(filename)["bearing"][0][0][2])     #Take out the data

    data=[] 
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start +=signal_size
        end +=signal_size
    return data,lab

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

def train_test_split_order(data_pd, test_size=0.8, num_classes=10):
    train_pd = pd.DataFrame(columns=('data', 'label'))
    val_pd = pd.DataFrame(columns=('data', 'label'))
    for i in range(num_classes):
        data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True)
        train_pd = train_pd.append(data_pd_tmp.loc[:int((1-test_size)*data_pd_tmp.shape[0]), ['data', 'label']], ignore_index=True)
        val_pd = val_pd.append(data_pd_tmp.loc[int((1-test_size)*data_pd_tmp.shape[0]):, ['data', 'label']], ignore_index=True)
    return train_pd,val_pd

#--------------------------------------------------------------------------------------------------------------------
class MFPT(object):
    num_classes = 15
    inputchannel = 1

    def __init__(self, data_dir,normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_preprare(self, test=False):
        list_data = get_files(self.data_dir, test)
        if test:
            test_dataset = dataset(list_data=list_data, test=True, transform=None)
            return test_dataset
        else:
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split_order(data_pd, test_size=0.2, num_classes= 15)
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train',self.normlizetype))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val',self.normlizetype))
            return train_dataset, val_dataset
