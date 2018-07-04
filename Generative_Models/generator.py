import torch
import torch.nn as nn

import torch.nn.functional as F


def Generator(z_dim=62, dataset='mnist', conditional=False, model='VAE'):

    return MNIST_Generator(z_dim, dataset, conditional, model)


class MNIST_Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, z_dim=62, dataset='mnist', conditional=False, model='VAE'):
        super(MNIST_Generator, self).__init__()
        self.dataset = dataset
        self.z_dim = z_dim
        self.model = model
        self.conditional = conditional

        self.input_height = 28
        self.input_width = 28
        self.input_dim = z_dim
        if self.conditional:
            self.input_dim += 10
        self.output_dim = 1

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )

        self.maxPool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.Sigmoid = nn.Sigmoid()
        self.apply(self.weights_init)

    def forward(self, input, c=None):

        if c is not None:
            input = input.view(-1, self.input_dim - 10)
            input = torch.cat([input, c], 1)
        else:
            input = input.view(-1, self.input_dim)

        x = self.fc(input)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)
        return x

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
