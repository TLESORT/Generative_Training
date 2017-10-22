import torch
import torch.nn as nn

class Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, z_dim=62, dataset='mnist', conditional=False, model='VAE'):
        super(Generator, self).__init__()
        self.dataset = dataset
        self.z_dim=z_dim
        self.model=model
        nz = 100
        if conditional:
            nz = nz + 10
        ngf = 64
        ndf = 64
        if self.dataset == 'mnist':
            nc = 1
        else:
            nc = 3

        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = z_dim
            if conditional:
                self.input_dim += 10
            self.output_dim = 1
        elif dataset == 'cifar10':
            self.input_height = 32
            self.input_width = 32
            self.input_dim = z_dim
            if conditional:
                self.input_dim += 10
            self.output_dim = 3
        elif dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = z_dim
            if conditional:
                self.input_dim += 10
            self.output_dim = 3

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

        if dataset == 'cifar10':
            self.ReLU = nn.ReLU(True)
            self.Tanh = nn.Tanh()
            self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
            self.BatchNorm1 = nn.BatchNorm2d(ngf * 8)

            self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
            self.BatchNorm2 = nn.BatchNorm2d(ngf * 4)

            self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
            self.BatchNorm3 = nn.BatchNorm2d(ngf * 2)

            self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False)
            self.BatchNorm4 = nn.BatchNorm2d(ngf * 1)

            self.conv5 = nn.ConvTranspose2d(ngf * 1, nc, 4, 2, 1, bias=False)

            self.apply(self.weights_init)

    def forward(self, input, c=None):
        if c is not None:
            input = torch.cat([input, c], 1)
        if self.dataset == 'cifar10':
            x = self.conv1(input.view(-1, self.input_dim, 1, 1))
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

            x = self.conv5(x)

            if self.model == 'VAE' or self.model == 'CVAE':
                x = self.Sigmoid(self.maxPool(x))
            else:
                x = self.Tanh(self.maxPool(x))
        else:
            x = self.fc(input.view(-1,self.input_dim))
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
