import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from load_dataset import load_dataset
import utils


from Generative_Model import GenerativeModel

# CGAN G(z)
class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.fc1_1 = nn.Linear(100, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(10, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 784)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(input)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = F.tanh(self.fc4(x))
        return x.view(-1, 1, 28, 28)

class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1_1 = nn.Linear(784, 1024)
        self.fc1_2 = nn.Linear(10, 1024)
        self.fc2 = nn.Linear(2048, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        input = input.view(-1, 784)
        x = F.leaky_relu(self.fc1_1(input), 0.2)
        y = F.leaky_relu(self.fc1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        x = F.sigmoid(self.fc4(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class CGAN(GenerativeModel):
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
        self.c = 0.01  # clipping value
        self.n_critic = 5  # the number of iterations of the critic per generator iteration
        self.conditional = args.conditional
        self.generators = []
        self.nb_batch = args.size_epoch
        self.num_examples = args.num_examples
        self.c_criterion = nn.NLLLoss()

        self.device = args.device
        # networks init

        # load dataset
        data_loader = load_dataset(self.dataset, self.batch_size, self.num_examples)
        self.data_loader_train = data_loader[0]
        self.data_loader_valid = data_loader[1]

        if self.dataset == 'mnist':
            self.z_dim = 100
            self.input_size = 1
            self.size = 28
        elif self.dataset == 'fashion-mnist':
            self.z_dim = 100
            self.input_size = 1
            self.size = 28

        elif self.dataset == 'cifar10':
            self.input_size = 3
            self.size = 32
            self.imageSize=32
            self.z_dim = 100

        elif self.dataset == 'celebA':
            self.z_dim = 100

        # Define network
        self.G = generator()
        self.D = discriminator()
        self.G.weight_init(mean=0, std=0.02)
        self.D.weight_init(mean=0, std=0.02)
        self.G.cuda()
        self.D.cuda()

        lr = 0.0002
        # Adam optimizer
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))

        # fixed noise
        if self.gpu_mode:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)).cuda(), volatile=True)
        else:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)), volatile=True)

    def train_all_classes(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        temp_z_ = torch.rand(10, 100)
        fixed_z_ = temp_z_
        fixed_y_ = torch.zeros(10, 1)
        for i in range(9):
            fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
            temp = torch.ones(10,1) + i
            fixed_y_ = torch.cat([fixed_y_, temp], 0)

        fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)
        fixed_y_label_ = torch.zeros(100, 10)
        fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
        fixed_y_label_ = Variable(fixed_y_label_.cuda(), volatile=True)

        # training parameters
        batch_size = 8
        lr = 0.0002
        train_epoch = 47

        # Binary Cross Entropy loss
        BCE_loss = nn.BCELoss()

        print('training start!')
        start_time = time.time()
        for epoch in range(train_epoch):
            self.G.train()
            D_losses = []
            G_losses = []

            # learning rate decay
            """
            if (epoch+1) == 30:
                self.G_optimizer.param_groups[0]['lr'] /= 10
                self.D_optimizer.param_groups[0]['lr'] /= 10
                print("learning rate change!")

            if (epoch+1) == 40:
                self.G_optimizer.param_groups[0]['lr'] /= 10
                self.D_optimizer.param_groups[0]['lr'] /= 10
                print("learning rate change!")
            """
            epoch_start_time = time.time()
            for x_, y_ in self.data_loader_train:
                # train discriminator D
                self.D.zero_grad()

                mini_batch = x_.size()[0]

                y_real_ = torch.ones(mini_batch)
                y_fake_ = torch.zeros(mini_batch)
                y_label_ = torch.zeros(mini_batch, 10)
                y_label_.scatter_(1, y_.view(mini_batch, 1), 1)

                x_ = x_.view(-1, 28 * 28)
                x_, y_label_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_label_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
                D_result = self.D(x_, y_label_).squeeze()
                D_real_loss = BCE_loss(D_result, y_real_)

                z_ = torch.rand((mini_batch, 100))
                y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor)
                y_label_ = torch.zeros(mini_batch, 10)
                y_label_.scatter_(1, y_.view(mini_batch, 1), 1)

                z_, y_label_ = Variable(z_.cuda()), Variable(y_label_.cuda())
                G_result = self.G(z_, y_label_)

                D_result = self.D(G_result, y_label_).squeeze()
                D_fake_loss = BCE_loss(D_result, y_fake_)
                D_fake_score = D_result.data.mean()

                D_train_loss = D_real_loss + D_fake_loss

                D_train_loss.backward()
                self.D_optimizer.step()

                D_losses.append(D_train_loss.data[0])

                # train generator G
                self.G.zero_grad()

                z_ = torch.rand((mini_batch, 100))
                y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor)
                y_label_ = torch.zeros(mini_batch, 10)
                y_label_.scatter_(1, y_.view(mini_batch, 1), 1)

                z_, y_label_ = Variable(z_.cuda()), Variable(y_label_.cuda())

                G_result = self.G(z_, y_label_)
                D_result = self.D(G_result, y_label_).squeeze()
                G_train_loss = BCE_loss(D_result, y_real_)
                G_train_loss.backward()
                self.G_optimizer.step()

                G_losses.append(G_train_loss.data[0])

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time


            print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                    torch.mean(torch.FloatTensor(G_losses))))
            self.train_hist['D_loss'].append(torch.mean(torch.FloatTensor(D_losses)))
            self.train_hist['G_loss'].append(torch.mean(torch.FloatTensor(G_losses)))
            self.train_hist['per_epoch_time'].append(per_epoch_ptime)

            self.save()
            self.visualize_results((epoch + 1))





