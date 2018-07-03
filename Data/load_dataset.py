from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.utils import data

from Data.input_pipeline import get_image_folders, get_test_image_folders
import numpy as np
import utils

import torch

from Data.fashion import fashion



class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def load_dataset_full(dataset, num_examples=50000):

    fas=False
    path = './Data/Datasets/'

    if dataset == 'mnist':
        dataset = datasets.MNIST(path + 'mnist', train=True, download=True, transform=transforms.ToTensor())
        dataset_train = Subset(dataset, range(num_examples))
        dataset_val = Subset(dataset, range(50000, 60000))
    elif dataset == 'fashion-mnist':
        if fas:
            dataset = datasets.FashionMNIST(path + 'fashion-mnist', train=True, download=True, transform=transforms.ToTensor())
        else:
            dataset = fashion(path + 'fashion-mnist', train=True, download=True, transform=transforms.ToTensor())
        dataset_train = Subset(dataset, range(num_examples))
        dataset_val = Subset(dataset, range(50000, 60000))

    list_classes_train = np.asarray([dataset_train[i][1] for i in range(len(dataset_train))])
    list_classes_val = np.asarray([dataset_val[i][1] for i in range(len(dataset_val))])

    return dataset_train, dataset_val, list_classes_train, list_classes_val



def load_dataset_test(dataset, batch_size):
    list_classes_test = []

    fas=False

    path = './Data/Datasets/'
    
    if dataset == 'mnist':
        dataset_test = datasets.MNIST(path + 'mnist', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset == 'fashion-mnist':
        if fas:
            dataset_test = DataLoader(
                datasets.FashionMNIST(path + 'fashion-mnist', train=False, download=True, transform=transforms.Compose(
                    [transforms.ToTensor()])),
                batch_size=batch_size)
        else:
            dataset_test = fashion(path + 'fashion-mnist', train=False, download=True, transform=transforms.ToTensor())

    list_classes_test = np.asarray([dataset_test[i][1] for i in range(len(dataset_test))])

    return dataset_test, list_classes_test


def get_iter_dataset(dataset, list_classe=[], batch_size=64, classe=None):
    if classe is not None:
        dataset = Subset(dataset, np.where(list_classe == classe)[0])

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader
