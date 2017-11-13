import numpy as np
import torch
import os
from torch.autograd import Variable
import utils
import copy
import GAN


def get_list_batch(train_loader):
    list_digits = [[], [], [], [], [], [], [], [], [], []]
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size = len(target)
        for i in range(batch_size):
            list_digits[target[i]].append(data[i])
    return list_digits


def get_batch(list_digits, digit, batch_size):
    liste = list_digits[digit]
    size_list = len(liste)
    batch = torch.zeros((batch_size, liste[0].shape[0], liste[0].shape[1], liste[0].shape[2]))
    for i in range(batch_size):
        indice = np.random.randint(0, size_list)
        batch[i] = liste[indice]
    return batch


def add_noise(batch):
    return Variable(batch.data + 0.1 * torch.randn(batch.data.shape))


'''
mask_idx = torch.nonzero(t_ == classe)
if mask_idx.dim() == 0:
    continue
x_ = torch.index_select(x_, 0, mask_idx[:, 0])
'''
