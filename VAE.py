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

import copy

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x).cuda()

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    KLD = torch.mean(KLD)
    KLD /= 784
    KLD = KLD.cuda()
    return BCE + KLD#, KLD, BCE

'''
class Generator(nn.Module):
    def __init__(self, z_dim=62, dataset='mnist', conditional=False):
        super(Generator, self).__init__()
        self.z_dim=z_dim
        self.fc1 = nn.Linear(3 * 1024, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(self.z_dim, 400)
        self.fc4 = nn.Linear(400, 3 * 1024)

        self.fc5 = nn.Linear(20, 400)  # can eventually share parameters with fc3
        self.fc6 = nn.Linear(400, 10)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        return self.decode(x.view(-1, self.z_dim))
'''
class Encoder(nn.Module):
    def __init__(self, z_dim, dataset='mnist', conditional=False):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.conditional = conditional
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_size = 784
        elif dataset == 'celebA':
            self.input_size = 64 * 64 * 3
        elif dataset == 'cifar10':
            self.input_size = 32 * 32 * 3
            #self.input_size = 64 * 64 * 3
        if self.conditional:
            self.input_size += 10
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(self.input_size, 1200)
        self.fc21 = nn.Linear(1200, z_dim)
        self.fc22 = nn.Linear(1200, z_dim)

    def encode(self, x, c=None):
        if self.conditional:
            x = torch.cat([x, c], 1)
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar, cuda=True):
        std = logvar.mul(0.5).exp_()
        if cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x, c=None):
        mu, logvar = self.encode(x.view(x.size()[0], -1), c)
        z = self.reparametrize(mu, logvar)
        return z.view(x.size()[0], self.z_dim, 1, 1), mu, logvar


class VAE(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 64
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.conditional = args.conditional
        self.gpu_mode = args.gpu_mode
        self.device = args.device
        self.nb_batch = args.nb_batch
        self.generators = []


        if self.conditional:
            self.model_name = 'C' + self.model_name

        # load dataset
        self.data_loader = load_dataset(self.dataset, self.batch_size)
        if self.dataset == 'mnist':
            self.z_dim = 20
            self.input_size = 1
            self.size = 28
        elif self.dataset == 'fashion-mnist':
            self.z_dim = 62
            self.input_size = 1
            self.size = 28

        elif self.dataset == 'cifar10':
            self.input_size = 3
            self.size = 32
            self.imageSize=32
            self.z_dim = 100

        elif self.dataset == 'celebA':
            self.z_dim = 100

        self.E = Encoder(self.z_dim, self.dataset, self.conditional)
        self.G = Generator(self.z_dim, self.dataset, self.conditional)
        self.E_optimizer = optim.Adam(self.E.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.E.cuda(self.device)
            self.G.cuda(self.device)

        # fixed noise
        if self.gpu_mode:
            self.sample_z_ = Variable(torch.randn((self.batch_size, self.z_dim,1,1)).cuda(self.device), volatile=True)
        else:
            self.sample_z_ = Variable(torch.randn((self.batch_size, self.z_dim,1,1)), volatile=True)


    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x).cuda()

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        KLD = torch.mean(KLD)
        KLD /= 784
        KLD = KLD.cuda()
        return BCE + KLD#, KLD, BCE


    def train_all_classes(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
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
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for tour in range(self.size_epoch):
                for iter, (x_, t_) in enumerate(self.data_loader):

                    if iter == self.data_loader.dataset.__len__() // self.batch_size:
                        break

                    if self.conditional:
                        y_onehot = torch.FloatTensor(t_.shape[0], 10)
                        y_onehot.zero_()
                        y_onehot.scatter_(1, t_[:, np.newaxis], 1.0)
                    else:
                        y_onehot = None
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
                    self.train_hist['D_loss'].append(G_loss.data[0])  # sake of simplicity
                    self.train_hist['G_loss'].append(G_loss.data[0])
                    G_loss.backward(retain_variables=True)

                    self.E_optimizer.step()
                    self.G_optimizer.step()

                    if ((iter + 1) % 100) == 0:
                        print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                              ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size,
                               G_loss.data[0], G_loss.data[0]))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.visualize_results((epoch + 1))
            self.save()

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def train(self):

        list_classes = sort_utils.get_list_batch(self.data_loader, self.nb_batch)  # list filled all classe sorted by class
        print(' training start!! (no conditional)')
        start_time = time.time()
        for classe in range(10):
            self.train_hist = {}
            self.train_hist['D_loss'] = []
            self.train_hist['G_loss'] = []
            self.train_hist['per_epoch_time'] = []
            self.train_hist['total_time'] = []
            for epoch in range(self.epoch):

                epoch_start_time = time.time()
                for iter in range(self.nb_batch):

                    x_ = sort_utils.get_batch(list_classes, classe, self.batch_size)
                    x_ = torch.FloatTensor(x_)
                    x_ = Variable(x_)
                    if self.gpu_mode:
                        x_ = x_.cuda(self.device)

                    # VAE
                    z_, mu, logvar = self.E(x_)
                    recon_batch = self.G(z_)

                    # train
                    self.G_optimizer.zero_grad()
                    self.E_optimizer.zero_grad()
                    g_loss = self.loss_function(recon_batch, x_, mu, logvar)
                    g_loss.backward(retain_variables=True)
                    self.G_optimizer.step()
                    self.E_optimizer.step()

                    self.train_hist['D_loss'].append(g_loss.data[0])
                    self.train_hist['G_loss'].append(g_loss.data[0])

                    if ((iter + 1) % 100) == 0:
                        print("classe : [%1d] Epoch: [%2d] [%4d/%4d] G_loss: %.8f, E_loss: %.8f" %
                              (classe, (epoch + 1), (iter + 1), self.nb_batch, g_loss.data[0], g_loss.data[0]))

                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                self.visualize_results((epoch + 1), classe)
                self.save_G(classe)
            result_dir=self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + 'classe-' + str(classe)
            utils.generate_animation(result_dir + '/' + self.model_name, self.epoch)
            utils.loss_plot(self.train_hist, result_dir, self.model_name)

            np.savetxt(
                os.path.join(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + 'classe-' + str(
                    classe), 'vae_training_' + self.dataset + '.txt'),
                np.transpose([self.train_hist['G_loss']]))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    def save_G(self, classe):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '-' + str(classe) + '_G.pkl'))

    def visualize_results(self, epoch, classe=None, fix=True):
        self.G.eval()
        dir_path = self.result_dir + '/' + self.dataset + '/' + self.model_name
        if classe is not None:
            dir_path = self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + 'classe-' + str(classe)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        if self.conditional:
            y = torch.LongTensor(self.batch_size, 1).random_() % 10
            y_onehot = torch.FloatTensor(self.batch_size, 10)
            y_onehot.zero_()
            y_onehot.scatter_(1, y, 1.0)
            y_onehot = Variable(y_onehot.cuda(self.device))
        else:
            y_onehot = None
        if fix:
            """ fixed noise """
            if self.conditional:
                samples = self.G(self.sample_z_, y_onehot)
            else:
                samples = self.G(self.sample_z_)
        else:
            """ random noise """
            if self.gpu_mode:
                sample_z_ = Variable(torch.randn((self.batch_size, self.z_dim, 1, 1)).cuda(self.device), volatile=True)
            else:
                sample_z_ = Variable(torch.randn((self.batch_size, self.z_dim, 1, 1)), volatile=True)

            if self.conditional:
                samples = self.G(sample_z_, y_onehot)
            else:
                samples = self.G(self.sample_z)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          dir_path + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    # Get samples and label from CVAE and VAE
    def sample(self, batch_size, classe=None):
        self.G.eval()
        if self.conditional:
            z_ = torch.randn(batch_size, self.z_dim,1,1)
            if self.gpu_mode:
                z_ = z_.cuda(self.device)
            if classe is not None:
                y = torch.ones(batch_size, 1) * classe
            else:
                y = torch.LongTensor(batch_size, 1).random_() % 10
            y_onehot = torch.FloatTensor(batch_size, 10)
            y_onehot.zero_()
            y_onehot.scatter_(1, y, 1.0)
            y_onehot = Variable(y_onehot.cuda(self.device))
            output = self.G(Variable(z_), y_onehot).data
        else:
            z_ = torch.randn(self.batch_size,1, self.z_dim, 1, 1)
            if self.gpu_mode:
                z_ = z_.cuda(self.device)
            y = (torch.randperm(1000) % 10)[:batch_size]
            output = torch.FloatTensor(batch_size, self.input_size, self.size, self.size)
            if classe is not None:
                output = self.generators[classe](Variable(z_))
            else:
                for i in range(batch_size):
                    classe = int(y[i])
                    output[i] = self.generators[classe](Variable(z_[i])).data.cpu()
                if self.gpu_mode:
                    output = output.cuda(self.device)
        return output, y

    def load_generators(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
        paths = [x for x in os.listdir(save_dir) if x.endswith("_G.pkl")]
        paths.sort()
        self.generators = []
        for i in range(10):
            model_path = os.path.join(save_dir, paths[i])
            self.G.load_state_dict(torch.load(model_path))
            if self.gpu_mode:
                self.generators.append(copy.deepcopy(self.G.cuda(self.device)))
            else:
                self.generators.append(copy.deepcopy(self.G))

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.E.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_E.pkl')))

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.E.state_dict(), os.path.join(save_dir, self.model_name + '_E.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)
