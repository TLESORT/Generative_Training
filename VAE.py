from __future__ import print_function
import argparse
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

def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD

class decoder(nn.Module):
    def __init__(self, dataset = 'mnist'):
        super(decoder, self).__init__()	
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

class encoder(nn.Module)
    def __init__(self, dataset = 'mnist'):
        super(encoder, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

	def encode(self, x):
		h1 = self.relu(self.fc1(x))
		return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
	    std = logvar.mul(0.5).exp_()
	    if cuda:
	        eps = torch.cuda.FloatTensor(std.size()).normal_()
	    else:
	        eps = torch.FloatTensor(std.size()).normal_()
	    eps = Variable(eps)
	    return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

class VAE(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 16
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type

        Enc=encoder()
        Dec=decoder()



    def train(epoch):
        size_epoch=2000
        model.train()
        train_loss = 0
        train_loss_classif = 0
        digit=(epoch-1)%10
        target = Variable(torch.LongTensor(batch_size).fill_(digit))

        for batch_idx in range(size_epoch):
            data=get_batch(list_digits, digit, batch_size)
            data=torch.FloatTensor(data)
            data = Variable(data)
            if cuda:
                data = data.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            # VAE
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward(retain_variables=True)
            train_loss += loss.data[0]

            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), size_epoch,
                    100. * batch_idx / size_epoch,
                    loss.data[0] / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / size_epoch))
        print('====> Epoch: {} Average loss classif: {:.4f}'.format(
              epoch, train_loss_classif / size_epoch))
