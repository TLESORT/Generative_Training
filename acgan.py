"""""""""
Pytorch implementation of Conditional Image Synthesis with Auxiliary Classifier GANs (https://arxiv.org/pdf/1610.09585.pdf).
This code is based on Deep Convolutional Generative Adversarial Networks in Pytorch examples : https://github.com/pytorch/examples/tree/master/dcgan
"""""""""
from __future__ import print_function
import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn as nn
import utils
import sort_utils
import pickle
from load_dataset import load_dataset
from generator import Generator

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class generator(nn.Module):

    def __init__(self, nz, ngf, nc):

        super(generator, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.Tanh = nn.Tanh()
        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(ngf * 8)

        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ngf * 4)

        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ngf * 2)

        self.conv4 = nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(nc)

        self.conv5 = nn.ConvTranspose2d(ngf * 1, nc, 4, 2, 1, bias=False)
        self.linear = nn.Linear(nc*32*32, 28*28)

        self.apply(weights_init)


    def forward(self, input):

        x = self.conv1(input)
        x = self.BatchNorm1(x)
        x = self.ReLU(x)

        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.ReLU(x)

        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.ReLU(x)

        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.ReLU(x)
        #x = self.conv5(x)
        x = self.linear(x.view(-1, 32*32))
        output = self.Tanh(x).view(-1, 1, 28, 28)
        return output

class discriminator(nn.Module):

    def __init__(self, ndf, nc, nb_label):

        super(discriminator, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 1, 4, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 1, 4, 1, 0, bias=False)
        self.disc_linear = nn.Linear(ndf * 1, 1)
        self.aux_linear = nn.Linear(ndf * 1, nb_label)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.ndf = ndf
        self.apply(weights_init)

    def forward(self, input):

        x = self.conv1(input)
        x = self.LeakyReLU(x)

        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.LeakyReLU(x)

        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.LeakyReLU(x)

        x = self.conv4(x)
        #x = self.BatchNorm4(x)
        #x = self.LeakyReLU(x)

        #x = self.conv5(x)
        x = x.view(-1, self.ndf * 1)
        c = self.aux_linear(x)
        c = self.softmax(c)
        s = self.disc_linear(x)
        s = self.sigmoid(s)
        return s,c

class ACGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 16
        self.batch_size = args.batch_size
        self.batchsize = self.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.conditional = args.conditional
        self.device = args.device
        self.num_examples = args.num_examples

        nz = 100
        ngf = 32
        ndf = 16
        self.imageSize = 32
        if self.dataset == 'mnist' or self.dataset == 'fashion-mnist':
            nc = 1
            nb_label = 10
            self.imageSize = 28
        else:
            nc = 3
            nb_label = 10
        self.nz = nz
        self.nc = nc
        # load dataset
        self.data_loader = load_dataset(self.dataset, self.batch_size)
        self.data_loader_train = self.data_loader[0]
        self.data_loader_valid = self.data_loader[1]
        # networks init
        # self.netG = generator(nz, ngf, nc)
        self.netG = Generator(nz, self.dataset, model='ACGAN')
        self.netD = discriminator(ndf, nc, nb_label)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        if self.gpu_mode:
            self.netD.cuda()
            self.netG.cuda()

        print('---------- Networks architecture -------------')
        utils.print_network(self.netG)
        utils.print_network(self.netD)
        print('-----------------------------------------------')


    def train_all_classes(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        input = torch.FloatTensor(self.batchsize, self.nc, self.imageSize, self.imageSize)
        noise = torch.FloatTensor(self.batchsize, self.nz, 1, 1)
        fixed_noise = torch.FloatTensor(self.batchsize, self.nz, 1, 1).normal_(0, 1)
        s_label = torch.FloatTensor(self.batchsize)
        c_label = torch.LongTensor(self.batchsize)
        s_criterion = nn.BCELoss()
        c_criterion = nn.NLLLoss()
        real_label = 1
        fake_label = 0
        nb_label = 10
        batch_size = self.batch_size
        nz = self.nz

        if self.gpu_mode:
            s_criterion.cuda()
            c_criterion.cuda()
            input, s_label = input.cuda(), s_label.cuda()
            c_label = c_label.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        input = Variable(input)
        s_label = Variable(s_label)
        c_label = Variable(c_label)
        noise = Variable(noise)
        fixed_noise = Variable(fixed_noise)
        fixed_noise_ = np.random.normal(0, 1, (self.batchsize, self.nz))
        random_label = np.random.randint(0, nb_label, self.batchsize)
        print('fixed label:{}'.format(random_label))
        random_onehot = np.zeros((self.batchsize, nb_label))
        random_onehot[np.arange(self.batchsize), random_label] = 1
        fixed_noise_[np.arange(self.batchsize), :nb_label] = random_onehot[np.arange(self.batchsize)]

        fixed_noise_ = (torch.from_numpy(fixed_noise_))
        fixed_noise_ = fixed_noise_.resize_(self.batchsize, self.nz, 1, 1)
        fixed_noise.data.copy_(fixed_noise_)

        dir_path = self.result_dir + '/' + self.dataset + '/' + self.model_name
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        def test(predict, labels):
            correct = 0
            pred = predict.data.max(1)[1]
            correct = pred.eq(labels.data).cpu().sum()
            return correct, len(labels.data)

        best = 1000000
        for epoch in range(self.epoch):
            for i, data in enumerate(self.data_loader_train, 0):
                ###########################
                # (1) Update D network
                ###########################
                # train with real
                self.netD.zero_grad()
                img, label = data
                batch_size = img.size(0)
                input.data.resize_(img.size()).copy_(img)
                s_label.data.resize_(batch_size).fill_(real_label)
                c_label.data.resize_(batch_size).copy_(label)
                s_output, c_output = self.netD(input)
                s_errD_real = s_criterion(s_output, s_label)
                c_errD_real = c_criterion(c_output, c_label)
                errD_real = s_errD_real + c_errD_real
                errD_real.backward()
                D_x = s_output.data.mean()
                
                correct, length = test(c_output, c_label)

                # train with fake
                noise.data.resize_(batch_size, nz, 1, 1)
                noise.data.normal_(0, 1)

                label = np.random.randint(0, nb_label, batch_size)
                noise_ = np.random.normal(0, 1, (batch_size, nz))
                label_onehot = np.zeros((batch_size, nb_label))
                label_onehot[np.arange(batch_size), label] = 1
                noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]
                
                noise_ = (torch.from_numpy(noise_))
                noise_ = noise_.resize_(batch_size, nz, 1, 1)
                noise.data.copy_(noise_)

                c_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

                fake = self.netG(noise)
                s_label.data.fill_(fake_label)
                s_output,c_output = self.netD(fake.detach())
                s_errD_fake = s_criterion(s_output, s_label)
                c_errD_fake = c_criterion(c_output, c_label)
                errD_fake = s_errD_fake + 0.01 * c_errD_fake

                errD_fake.backward()
                D_G_z1 = s_output.data.mean()
                errD = s_errD_real + s_errD_fake
                self.optimizerD.step()

                ###########################
                # (2) Update G network
                ###########################
                self.netG.zero_grad()
                s_label.data.fill_(real_label)  # fake labels are real for generator cost
                s_output,c_output = self.netD(fake)
                s_errG = s_criterion(s_output, s_label)
                c_errG = c_criterion(c_output, c_label)
                
                errG = s_errG + c_errG
                errG.backward()
                D_G_z2 = s_output.data.mean()
                self.optimizerG.step()


            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f, Accuracy: %.4f / %.4f = %.4f'
                    % (epoch, self.epoch, i, len(self.data_loader_train),
                        errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2,
                        correct, length, 100.* correct / length))
            # do checkpointing
            if epoch % 5 == 0:
                self.save()
                self.visualize_results((epoch + 1), fixed_noise)

    # Get samples and label from CVAE
    def sample(self, batch_idx):
        self.netG.eval()
        noise = torch.FloatTensor(self.batchsize, self.nz, 1, 1)
        noise = noise.cuda()
        noise = Variable(noise)

        label = np.random.randint(0, 10, self.batch_size)
        noise_ = np.random.normal(0, 1, (self.batch_size, self.nz))
        label_onehot = np.zeros((self.batch_size, 10))
        label_onehot[np.arange(self.batch_size), label] = 1
        noise_[np.arange(self.batch_size), :10] = label_onehot[np.arange(self.batch_size)]

        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(self.batch_size, self.nz, 1, 1)
        noise.data.copy_(noise_)
        output = self.netG(noise)
        return output.data, torch.LongTensor(label)

    def visualize_results(self, epoch, fixed_noise, classe=None, fix=True):
        self.netG.eval()
        dir_path = self.result_dir + '/' + self.dataset + '/' + self.model_name + '/num_examples_' + str(self.num_examples)
        if classe is not None:
            dir_path = self.result_dir + '/' + self.dataset + '/' + self.model_name + '/num_examples_' +\
                    str(self.num_examples) + '/' + 'classe-' + str(classe)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        samples = self.netG(fixed_noise)
        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                dir_path + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name, 'num_examples_' + str(self.num_examples))

        self.netG.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.netD.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name, 'num_examples_' + str(self.num_examples))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.netG.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.netD.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

