import numpy as np
import torch
import os
from torch.autograd import Variable
import utils
import copy
import GAN


def get_sorted_dataset(dataset_loader, batch_size):
    # Getting all datas
    for i, (d, t) in enumerate(dataset_loader):
        if i == 0:
            datas = d
            labels = t
        else:
            datas = torch.cat((datas, d))
            labels = torch.cat((labels, t))
    # Sort by labels
    indices = np.argsort(labels.numpy())
    # Sort data according to new indices
    target_sorted = torch.index_select(labels, 0, torch.LongTensor(indices))
    data_sorted = torch.index_select(datas, 0, torch.LongTensor(indices))
    new_dataset = torch.utils.data.TensorDataset(data_sorted, target_sorted)
    data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size)
    return data_loader


def get_list_batch(train_loader,nb_batch):
    list_digits=[[],[],[],[],[],[],[],[],[],[]]
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > nb_batch:
            break  # make us control how many batch we use
        batch_size=len(target)
        for i in range(batch_size):
            #list_digits[target[i]].append(batch_idx*batch_size+i)
            list_digits[target[i]].append(data[i])

    return list_digits


def get_batch(list_digits, digit, batch_size):
    liste = list_digits[digit]
    size_list=len(liste)
    batch=torch.zeros((batch_size,liste[0].shape[0],liste[0].shape[1],liste[0].shape[2]))
    for i in range(batch_size):
        indice=np.random.randint(0, size_list)
        batch[i]=liste[indice]
    return batch

def add_noise(batch):
    return Variable(batch.data + 0.1*torch.randn(batch.data.shape))
