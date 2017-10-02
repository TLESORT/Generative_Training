import numpy as np
import torch
import os
from torch.autograd import Variable
import utils
import copy
import GAN

def get_list_batch(train_loader):
    list_digits=[[],[],[],[],[],[],[],[],[],[]]
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size=len(target)
        for i in range(batch_size):
            #list_digits[target[i]].append(batch_idx*batch_size+i)
            list_digits[target[i]].append(data[i])
    return list_digits

def get_batch(list_digits, digit, batch_size):
    liste=list_digits[digit]
    size_list=len(liste)
    batch=torch.zeros((batch_size,liste[0].shape[0],liste[0].shape[1],liste[0].shape[2]))
    for i in range(batch_size):
        indice=np.random.randint(0, size_list)
        batch[i]=liste[indice]
    return batch


def get_generators(path):
    paths = [x for x in os.listdir(path) if x.endswith("_G.pkl")]
    paths.sort()
    generators=[]
    for i in range(10):
        model_path=os.path.join(path, paths[i])
        G = GAN.generator(dataset = 'mnist')
        G.load_state_dict(torch.load(model_path))
        generators.append(G.cuda())
    return  generators

def get_generators_batch(generators,batch_size,noise):
    gene_indice=(torch.randperm(1000)%10)[:batch_size]
    batch=torch.FloatTensor(batch_size,1,28,28)
    target=torch.LongTensor(batch_size)
    for i in range(batch_size):
        target[i]=int(gene_indice[i])
        gene=generators[target[i]]
        h=Variable(noise[i].cuda())
        batch[i]=gene(h).data.cpu()
    return batch, target

