from __future__ import print_function
import argparse
import utils
import time
import os
import pickle
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms
from fashion import fashion
from torch.utils import data
import matplotlib.pyplot as plt
import sort_utils
from torch.utils.data import DataLoader

from generator import Generator
from load_dataset import load_dataset
from Generative_Model import GenerativeModel

import copy


class VAE(GenerativeModel):
    def loss_function(self, recon_x, x, mu, logvar):
        # BCE = F.binary_cross_entropy(recon_x, x).cuda()

        reconstruction_function = nn.BCELoss()
        reconstruction_function.size_average = False
        BCE = reconstruction_function(recon_x, x)



        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        KLD = torch.mean(KLD)
        KLD /= 784
        if self.gpu_mode:
            BCE = BCE.cuda()
            KLD = KLD.cuda()
        return BCE + KLD

    def train_all_classes(self):
        self.train_hist = {}
        self.train_hist['Train_loss'] = []
        self.train_hist['Valid_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.size_epoch = 1

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda(self.device)), Variable(
                torch.zeros(self.batch_size, 1).cuda(self.device))
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(
                torch.zeros(self.batch_size, 1))

        self.E.train()
        print('training start!!')
        start_time = time.time()
        best = 1000000
        early_stop = 0
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            sum_loss_train = 0.
            n_batch = 0.
            for iter, (x_, t_) in enumerate(self.data_loader_train):
                y_onehot = torch.FloatTensor(t_.shape[0], 10)
                y_onehot.zero_()
                y_onehot.scatter_(1, t_[:, np.newaxis], 1.0)
                if self.gpu_mode:
                    x_ = Variable(x_.cuda(self.device))
                    if self.conditional:
                        y_onehot = Variable(y_onehot.cuda(self.device))
                else:
                    x_ = Variable(x_)
                self.E_optimizer.zero_grad()
                self.G_optimizer.zero_grad()
                # VAE
                z_, mu, logvar = self.E(x_, y_onehot)
                recon_batch = self.G(z_, y_onehot)

                G_loss = self.loss_function(recon_batch, x_, mu, logvar)
                sum_loss_train += G_loss.data[0]
                G_loss.backward(retain_variables=True)

                self.E_optimizer.step()
                self.G_optimizer.step()
                n_batch += 1
                self.train_hist['Train_loss'].append(G_loss.data[0])

            sum_loss_train = sum_loss_train / np.float(n_batch)
            sum_loss_valid = 0.
            n_batch = 0.
            for iter, (x_, t_) in enumerate(self.data_loader_valid):
                n_batch += 1
                y_onehot = torch.FloatTensor(t_.shape[0], 10)
                y_onehot.zero_()
                y_onehot.scatter_(1, t_[:, np.newaxis], 1.0)
                if self.gpu_mode:
                    x_ = Variable(x_.cuda(self.device))
                    if self.conditional:
                        y_onehot = Variable(y_onehot.cuda(self.device))
                else:
                    x_ = Variable(x_)
                # VAE
                z_, mu, logvar = self.E(x_, y_onehot)
                recon_batch = self.G(z_, y_onehot)

                G_loss = self.loss_function(recon_batch, x_, mu, logvar)
                sum_loss_valid += G_loss.data[0]
                self.train_hist['Valid_loss'].append(G_loss.data[0])
            sum_loss_valid = sum_loss_valid / np.float(n_batch)
            print("Epoch: [%2d] Train_loss: %.8f, Valid_loss: %.8f" % ((epoch + 1), sum_loss_train, sum_loss_valid))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.visualize_results((epoch + 1))
            if sum_loss_valid < best:
                best = sum_loss_valid
                self.save()
                early_stop = 0.
            # We dit early stopping of the valid performance doesn't
            # improve anymore after 50 epochs
            if early_stop == 50:
                #break
                print("I should stop")
            else:
                early_stop += 1

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")
        result_dir = self.result_dir + '/' + self.dataset + '/' + self.model_name + '/num_examples_' + \
                     str(self.num_examples)
        utils.generate_animation(result_dir + '/' + self.model_name, self.epoch)
        # utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name,
        #                                               'num_examples_' + str(self.num_examples)), self.model_name)

        np.savetxt(
            os.path.join(result_dir + '/cvae_training_' +
                         self.dataset + '.txt'), np.transpose([self.train_hist['Train_loss']]))

    def train(self):

        list_classes = sort_utils.get_list_batch(self.data_loader_train)  # list filled all classe sorted by class
        list_classes_valid = sort_utils.get_list_batch(self.data_loader_valid)  # list filled all classe sorted by class
        print(' training start!! (no conditional)')
        start_time = time.time()
        for classe in range(10):
            self.train_hist = {}
            self.train_hist['D_loss'] = []
            self.train_hist['G_loss'] = []
            self.train_hist['per_epoch_time'] = []
            self.train_hist['total_time'] = []
            self.G.train()
            best = 100000
            early_stop = 0.
            for epoch in range(self.epoch):

                epoch_start_time = time.time()
                # print("number of batch data")
                # print(len(self.data_loader_train))

                sum_loss_train = 0.
                n_batch = 0.
                for iter in range(self.size_epoch):
                    #for iter, (x_, t_) in enumerate(self.data_loader_train):
                    x_ = sort_utils.get_batch(list_classes, classe, self.batch_size)
                    x_ = torch.FloatTensor(x_)
                    # Apply mask on the data to get the correct class
                    #mask_idx = torch.nonzero(t_ == classe)
                    #if mask_idx.dim() == 0:
                    #    continue
                    #x_ = torch.index_select(x_, 0, mask_idx[:, 0])
                    #t_ = torch.index_select(t_, 0, mask_idx[:, 0])
                    x_ = Variable(x_)
                    if self.gpu_mode:
                        x_ = x_.cuda(self.device)
                        #t_ = t_.cuda(self.device)
                    # VAE
                    z_, mu, logvar = self.E(x_)
                    recon_batch = self.G(z_)

                    # train
                    self.G_optimizer.zero_grad()
                    self.E_optimizer.zero_grad()
                    g_loss = self.loss_function(recon_batch, x_, mu, logvar)
                    sum_loss_train += g_loss.data[0]
                    g_loss.backward(retain_variables=True)
                    self.G_optimizer.step()
                    self.E_optimizer.step()

                    self.train_hist['D_loss'].append(g_loss.data[0])
                    self.train_hist['G_loss'].append(g_loss.data[0])

                    if ((iter + 1) % 100) == 0:
                        print("classe : [%1d] Epoch: [%2d] [%4d/%4d] G_loss: %.8f, E_loss: %.8f" %
                              (classe, (epoch + 1), (iter + 1), self.size_epoch, g_loss.data[0], g_loss.data[0]))
                    n_batch += 1
                sum_loss_train = sum_loss_train / np.float(n_batch)
                sum_loss_valid = 0.
                n_batch = 0.
                for iter, (x_, t_) in enumerate(self.data_loader_valid):
                    max_val, max_indice = torch.max(t_, 0)
                    mask_idx = torch.nonzero(t_ == classe)
                    if mask_idx.dim() == 0:
                        continue
                    x_ = torch.index_select(x_, 0, mask_idx[:, 0])
                    t_ = torch.index_select(t_, 0, mask_idx[:, 0])
                    if self.gpu_mode:
                        x_ = Variable(x_.cuda(self.device))
                    else:
                        x_ = Variable(x_)
                    # VAE
                    z_, mu, logvar = self.E(x_)
                    recon_batch = self.G(z_)

                    G_loss = self.loss_function(recon_batch, x_, mu, logvar)
                    sum_loss_valid += G_loss.data[0]
                    n_batch += 1
                sum_loss_valid = sum_loss_valid / np.float(n_batch)
                print("classe : [%1d] Epoch: [%2d] Train_loss: %.8f, Valid_loss: %.8f" % (
                classe, (epoch + 1), sum_loss_train, sum_loss_valid))
                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                self.visualize_results((epoch + 1), classe)
                if sum_loss_valid < best:
                    best = sum_loss_valid
                    self.save_G(classe)
                    early_stop = 0.
                # We dit early stopping of the valid performance doesn't
                # improve anymore after 50 epochs
                if early_stop == 50:
                    break
                else:
                    early_stop += 1
            result_dir = self.result_dir + '/' + self.dataset + '/' + self.model_name + '/num_examples_' + \
                         str(self.num_examples) + '/' + 'classe-' + str(classe)
            utils.generate_animation(result_dir + '/' + self.model_name, epoch + 1)
            utils.loss_plot(self.train_hist, result_dir, self.model_name)

            np.savetxt(
                os.path.join(result_dir, 'vae_training_' + self.dataset + '.txt'),
                np.transpose([self.train_hist['G_loss']]))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")
