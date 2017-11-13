import utils, torch, time, os, pickle
import sort_utils
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from fashion import fashion
from torch.utils import data
import copy
from generator import Generator
from load_dataset import load_dataset
from Generative_Model import GenerativeModel


'''
class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset='mnist', conditional=False):
        super(discriminator, self).__init__()
        self.dataset = dataset
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1
            self.output_dim = 1
        elif dataset == 'cifar10':
            self.input_height = 32
            self.input_width = 32
            self.input_dim = 62
            if conditional:
                self.input_dim += 10
            self.output_dim = 1
        elif dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 3
            self.output_dim = 1

        shape = 128 * (self.input_height // 4) * (self.input_width // 4)
        if conditional:
            shape += 10

        if dataset == 'cifar10':
            ndf = 32
            self.ndf = ndf
            self.conv = nn.Sequential(
                nn.Conv2d(3, ndf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 4, ndf* 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, ndf* 8, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                #nn.Sigmoid()
            )
            shape_fc = 0
            if conditional:
                shape_fc += 10
            self.fc = nn.Sequential(
                nn.Linear(ndf * 8 * 4 * 4 + shape_fc, self.output_dim),
                nn.Sigmoid(),
            )
        else:
            self.conv = nn.Sequential(
                    nn.Conv2d(self.input_dim, 64, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64, 128, 4, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2),
                    )
            self.fc = nn.Sequential(
                nn.Linear(shape, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, self.output_dim),
                nn.Sigmoid(),
            )
            utils.initialize_weights(self)

    def forward(self, input, c=None):
        if self.dataset == 'cifar10':
            x = self.conv(input)
            x = x.view(-1, 4 * 4 * self.ndf*8)
        else:
            x = self.conv(input)
            x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        if c is not None:
            x = torch.cat([x, c], 1)
        x = self.fc(x)
        return x
'''

class GAN(GenerativeModel):

    def train_all_classes(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda(self.device)), Variable(
                torch.zeros(self.batch_size, 1).cuda(self.device))
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(
                torch.zeros(self.batch_size, 1))

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, t_) in enumerate(self.data_loader_train):
                if iter == self.data_loader_train.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.conditional:
                    y_onehot = torch.FloatTensor(t_.shape[0], 10)
                    y_onehot.zero_()
                    y_onehot.scatter_(1, t_[:, np.newaxis], 1.0)
                else:
                    y_onehot = None
                if self.gpu_mode:
                    x_, z_ = Variable(x_.cuda(self.device)), Variable(z_.cuda(self.device))
                    if self.conditional:
                        y_onehot = Variable(y_onehot.cuda(self.device))
                else:
                    x_, z_ = Variable(x_), Variable(z_)

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_, y_onehot)
                D_real_loss = self.BCELoss()(D_real, self.y_real_)

                G_ = self.G(z_, y_onehot)
                D_fake = self.D(G_, y_onehot)
                D_fake_loss = self.BCELoss()(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.data[0])

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_onehot)
                D_fake = self.D(G_, y_onehot)
                G_loss = self.BCELoss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.data[0])

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                            ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size,
                            D_loss.data[0], G_loss.data[0]))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            if epoch % 50 == 0:
                self.visualize_results((epoch + 1))
                self.save()

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        utils.generate_animation(self.result_dir + '/' + self.model_name, self.epoch)
        utils.loss_plot(self.train_hist, self.save_dir, self.model_name)


    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.size_epoch = 1000

        list_classes = sort_utils.get_list_batch(self.data_loader_train)  # list filled all classe sorted by class

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda(self.device)), Variable(
                torch.zeros(self.batch_size, 1).cuda(self.device))
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(
                torch.zeros(self.batch_size, 1))

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for classe in range(10):
            for epoch in range(self.epoch):
                self.G.train()
                epoch_start_time = time.time()
                # for iter, (x_, _) in enumerate(self.data_loader):
                for iter in range(self.size_epoch):
                    x_ = sort_utils.get_batch(list_classes, classe, self.batch_size)
                    # if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    #    break

                    z_ = torch.rand((self.batch_size, self.z_dim))

                    if self.gpu_mode:
                        x_, z_ = Variable(x_.cuda(self.device)), Variable(z_.cuda(self.device))
                    else:
                        x_, z_ = Variable(x_), Variable(z_)

                    # update D network
                    self.D_optimizer.zero_grad()

                    D_real = self.D(x_)
                    D_real_loss = self.BCELoss(D_real, self.y_real_)

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    D_fake_loss = self.BCELoss(D_fake, self.y_fake_)

                    D_loss = D_real_loss + D_fake_loss
                    self.train_hist['D_loss'].append(D_loss.data[0])

                    D_loss.backward()
                    self.D_optimizer.step()

                    # update G network
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = self.BCELoss(D_fake, self.y_real_)
                    self.train_hist['G_loss'].append(G_loss.data[0])

                    G_loss.backward()
                    self.G_optimizer.step()

                    if ((iter + 1) % 100) == 0:
                        print("classe : [%1d] Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                              (classe, (epoch + 1), (iter + 1), self.size_epoch, D_loss.data[0], G_loss.data[0]))

                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                self.visualize_results((epoch + 1), classe)
                self.save_G(classe)
            utils.generate_animation(
                self.result_dir + '/' + 'classe-' + str(classe) + '/' + self.model_name, self.epoch)
            utils.loss_plot(self.train_hist, self.save_dir,self.model_name)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()

