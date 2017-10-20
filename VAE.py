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

class Encoder(nn.Module):
    def __init__(self, z_dim, dataset='mnist', conditional=False):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.conditional = conditional
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_size = 784
        if self.conditional:
            self.input_size += 10
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(self.input_size, 400)
        self.fc21 = nn.Linear(400, z_dim)
        self.fc22 = nn.Linear(400, z_dim)

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
        mu, logvar = self.encode(x.view(-1, 784), c)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar


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
        if self.conditional:
            self.model_name = 'C' + self.model_name

        self.z_dim = 20
        self.E = Encoder(self.z_dim, self.dataset, self.conditional)
        self.G = Generator(self.z_dim, self.dataset, self.conditional)
        self.E_optimizer = optim.Adam(self.E.parameters(), lr=args.lrD , betas=(args.beta1, args.beta2))
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG , betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.E.cuda(self.device)
            self.G.cuda(self.device)

        # load dataset
        if self.dataset == 'mnist':
            self.data_loader = DataLoader(datasets.MNIST('data/mnist', train=True, download=True,
                                                         transform=transforms.Compose(
                                                             [transforms.ToTensor()])),
                                          batch_size=self.batch_size, shuffle=True)
        elif self.dataset == 'fashion-mnist':
            # self.data_loader = DataLoader(
            #    datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transforms.Compose(
            #        [transforms.ToTensor()])),
            #    batch_size=self.batch_size, shuffle=True)

            kwargs = {'num_workers': 1, 'pin_memory': True} if self.gpu_mode else {}

            self.data_loader = data.DataLoader(
                fashion('fashion_data', train=True, download=True, transform=transforms.ToTensor()),
                batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

        elif self.dataset == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            trainset = datasets.CIFAR10(root='/Tmp/bordesfl/', train=True,
                                        download=True, transform=transform)
            self.data_loader = DataLoader(trainset, batch_size=self.batch_size,
                                          shuffle=True, num_workers=8)
            self.z_dim = 100
        elif self.dataset == 'celebA':
            self.data_loader = utils.load_celebA('data/celebA', transform=transforms.Compose(
                [transforms.CenterCrop(160), transforms.Scale(64), transforms.ToTensor()]), batch_size=self.batch_size,
                                                 shuffle=True)

        # fixed noise
        if self.gpu_mode:
            self.sample_z_ = Variable(torch.randn((self.batch_size, self.z_dim)).cuda(self.device), volatile=True)
        else:
            self.sample_z_ = Variable(torch.randn((self.batch_size, self.z_dim)), volatile=True)

    def train_all_classes(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.size_epoch = 2

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

                    G_loss = loss_function(recon_batch, x_, mu, logvar)
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
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.size_epoch = 1000
        self.E.train()
        self.G.train()

        list_classes = sort_utils.get_list_batch(self.data_loader)  # list filled all classe sorted by class
        print(' training start!! (no conditional)')
        start_time = time.time()
        for classe in range(10):
            for epoch in range(self.epoch):

                epoch_start_time = time.time()
                for iter in range(self.size_epoch):
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

                    g_loss = loss_function(recon_batch, x_, mu, logvar)
                    g_loss.backward(retain_variables=True)
                    self.G_optimizer.step()
                    self.E_optimizer.step()

                    self.train_hist['D_loss'].append(g_loss.data[0])
                    self.train_hist['G_loss'].append(g_loss.data[0])

                    if ((iter + 1) % 100) == 0:
                        print("classe : [%1d] Epoch: [%2d] [%4d/%4d] G_loss: %.8f, E_loss: %.8f" %
                              (classe, (epoch + 1), (iter + 1), self.size_epoch, g_loss.data[0], g_loss.data[0]))

                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                self.visualize_results((epoch + 1), classe)
                self.save_G(classe)

            utils.generate_animation(
                self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + 'classe-' + str(
                    classe) + '/' + self.model_name,
                self.epoch)
            utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name),
                            self.model_name)

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
                sample_z_ = Variable(torch.randn((self.batch_size, self.z_dim)).cuda(self.device), volatile=True)
            else:
                sample_z_ = Variable(torch.randn((self.batch_size, self.z_dim)), volatile=True)

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

    # Get samples and label from CVAE
    def sample(self, batch_idx):
        self.G.eval()
        z_ = torch.randn(self.batch_size, self.z_dim)
        z_ = Variable(z_.cuda())
        y = torch.LongTensor(batch_idx, 1).random_() % 10
        y_onehot = torch.FloatTensor(self.batch_size, 10)
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1.0)
        y_onehot = Variable(y_onehot.cuda())
        output = self.G(z_, y_onehot)
        return output, y

    def load_generators(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
        paths = [x for x in os.listdir(save_dir) if x.endswith("_G.pkl")]
        paths.sort()
        generators = []
        for i in range(10):
            model_path = os.path.join(save_dir, paths[i])
            self.G.load_state_dict(torch.load(model_path))
            if self.gpu_mode:
                generators.append(copy.deepcopy(self.G.cuda()))
            else:
                generators.append(copy.deepcopy(self.G))
        return generators

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
