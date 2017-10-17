import torch
import torch.nn as nn

class Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, z_dim=62, dataset='mnist', conditional=False):
        super(Generator, self).__init__()
        self.dataset = dataset
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

        if dataset == 'cifar10':
            ngf = 64
            self.ngf = ngf
            self.fc0 = nn.Linear(self.input_dim, 4*4*ngf*8)
            self.bn0 = nn.BatchNorm1d(4*4*ngf*8)
            self.relu0 = nn.ReLU(True)
            self.dcgan = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(ngf*8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf, 3, 3, 1, 1, bias=False),
                # nn.Sigmoid()
                #nn.BatchNorm2d(ngf),
                #nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                #nn.ConvTranspose2d(ngf,3, 3, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
            # utils.initialize_weights(self)

    def forward(self, input, c=None):
        if c is not None:
            input = torch.cat([input, c], 1)
        if self.dataset == 'cifar10':
            x = self.relu0(self.bn0(self.fc0(input)))
            x = x.view(-1, self.ngf * 8, 4, 4)
            x = self.dcgan(x)
        else:
            x = self.fc(input)
            x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
            x = self.deconv(x)
        return x
