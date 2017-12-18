import torch


def get_list_batch(train_loader):
    list_digits = [[], [], [], [], [], [], [], [], [], []]
    for batch_idx, (data, target) in enumerate(train_loader):
        for i in range(len(target)):
            list_digits[target[i]].append(data[i])
    return list_digits


def get_batch(list_digits, digit, batch_size):
    liste = list_digits[digit]
    size_list = len(liste)
    batch = torch.zeros((batch_size, liste[0].shape[0], liste[0].shape[1], liste[0].shape[2]))
    for i in range(batch_size):
        indice = torch.randperm(size_list)[0]
        batch[i] = liste[indice]
    return batch


def get_list_batch2(train_loader):
    list_digits_batch = [[], [], [], [], [], [], [], [], [], []]
    list_digits_ind = [[], [], [], [], [], [], [], [], [], []]
    for batch_idx, (data, target) in enumerate(train_loader):
        for i in range(len(target)):
            list_digits_batch[target[i]].append(batch_idx)
            list_digits_ind[target[i]].append(i)
    return list_digits_batch,list_digits_ind