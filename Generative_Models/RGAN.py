import utils, torch, time, os, pickle
import numpy as np
from torch.autograd import Variable

from Data.load_dataset import get_iter_dataset
from Generative_Models.Generative_Model import GenerativeModel

import torch
import torch.nn as nn
import torch.optim as optim


class Generator_RGAN(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, z_dim=62, dataset='mnist', conditional=False, model='VAE'):
        super(Generator_RGAN, self).__init__()
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
            nn.Sigmoid(), # No sigmoid for RGAN
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


class RGAN(GenerativeModel):
    def __init__(self, args):
        super(RGAN, self).__init__(args)
        self.c = 0.01  # clipping value
        self.n_critic = 5  # the number of iterations of the critic per generator iteration

        print("create G and D")
        self.G = Generator_RGAN(self.z_dim, self.dataset, self.conditional, self.model_name)

        print("create G and D 's optimizers")
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda(self.device)


    def train(self):
        self.G.apply(self.G.weights_init)
        print(' training start!! (no conditional)')
        start_time = time.time()



        for classe in range(10):
            self.train_hist = {}
            self.train_hist['D_loss'] = []
            self.train_hist['G_loss'] = []
            self.train_hist['per_epoch_time'] = []
            self.train_hist['total_time'] = []
            # self.G.apply(self.G.weights_init) does not work for instance
            self.G.train()

            data_loader_train = get_iter_dataset(self.dataset_train, self.list_class_train, self.batch_size, classe)

            print("Classe: " + str(classe))
            for epoch in range(self.epoch):

                epoch_start_time = time.time()

                for iter, (x_, t_) in enumerate(data_loader_train):

                    self.y_real_ = Variable(torch.ones(x_.size(0)))
                    self.y_fake_ = self.y_real_ * -1

                    if self.gpu_mode:
                        self.y_real_, self.y_fake_ = self.y_real_.cuda(self.device), self.y_fake_.cuda(self.device)

                    z_ = torch.randn((x_.size(0), self.z_dim, 1, 1))
                    if self.gpu_mode:
                        x_, z_ = Variable(x_.cuda(self.device)), Variable(z_.cuda(self.device))
                    else:
                        x_, z_ = Variable(x_), Variable(z_)


                    ##################################################

                    for p in self.D.parameters():
                        p.requires_grad = True

                    self.D_optimizer.zero_grad()

                    # Real data
                    y_pred = self.D(x_)
                    #y.data.resize_(current_batch_size).fill_(1)

                    # Fake data
                    #z.data.resize_(current_batch_size, param.z_size, 1, 1).normal_(0, 1)
                    fake = self.G(z_)
                    #x_fake.data.resize_(fake.data.size()).copy_(fake.data)



                    y_pred_fake = self.D(fake.detach())  # For generator step do not detach

                    y_pred_fake=y_pred_fake.view(x_.size(0))
                    y_pred=y_pred.view(x_.size(0))

                    #y2.data.resize_(current_batch_size).fill_(0)

                    # No activation in generator
                    BCE_stable = torch.nn.BCEWithLogitsLoss()

                    # Discriminator loss
                    D_loss = BCE_stable(y_pred - y_pred_fake, self.y_real_) # it was, I changed it to ...
                    #D_loss = BCE_stable(y_pred - y_pred_fake, self.y_fake_)
                    D_loss.backward()

                    self.D_optimizer.step()

                    for p in self.D.parameters():
                        p.requires_grad = False

                    self.G_optimizer.zero_grad()

                    # Generator loss (You may want to resample again from real and fake data)


                    ####### Resample (fake only) ##########
                    y_pred = self.D(x_)
                    fake = self.G(z_)
                    y_pred_fake = self.D(fake)  # For generator step do not detach

                    y_pred_fake = y_pred_fake.view(x_.size(0))
                    y_pred = y_pred.view(x_.size(0))

                    G_loss = BCE_stable(y_pred_fake - y_pred, self.y_real_)
                    G_loss.backward()

                    self.G_optimizer.step()

                    self.train_hist['G_loss'].append(G_loss.data[0])
                    self.train_hist['D_loss'].append(D_loss.data[0])

                    ##################################################


                    if ((iter + 1) % 100) == 0:
                        print("classe : [%1d] Epoch: [%2d] [%4d/%4d] G_loss: %.8f, D_loss: %.8f" %
                              (classe, (epoch + 1), (iter + 1), self.size_epoch, G_loss.data[0], D_loss.data[0]))
                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                self.visualize_results((epoch + 1), classe)
            self.save_G(classe)

            result_dir = self.result_dir + '/' + 'classe-' + str(classe)
            utils.generate_animation(result_dir + '/' + self.model_name, epoch+1)
            utils.loss_plot(self.train_hist, result_dir, self.model_name)

            np.savetxt(
                os.path.join(result_dir, 'wgan_training_' + self.dataset + '.txt'),
                np.transpose([self.train_hist['G_loss']]))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

