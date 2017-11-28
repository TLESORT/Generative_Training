import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sort_utils
from Generative_Model import GenerativeModel

from load_dataset import load_dataset
from generator import Generator
from discriminator import Discriminator


'''
class discriminator(nn.Module):
    # It must be Auto-Encoder style architecture
    # Architecture : (64)4c2s-FC32-FC64*14*14_BR-(1)4dc2s_S
    def __init__(self, dataset = 'mnist'):
        super(discriminator, self).__init__()
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1
            self.output_dim = 1
        elif dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 3
            self.output_dim = 3

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (self.input_height // 2) * (self.input_width // 2), 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64 * (self.input_height // 2) * (self.input_width // 2)),
            nn.BatchNorm1d(64 * (self.input_height // 2) * (self.input_width // 2)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            #nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = x.view(-1, 64, (self.input_height // 2), (self.input_width // 2))
        x = self.deconv(x)

        return x
'''


class BEGAN(GenerativeModel):
    def train(self):

        list_classes = sort_utils.get_list_batch(self.data_loader_train)

        for classe in range(10):
            self.train_hist = {}
            self.train_hist['D_loss'] = []
            self.train_hist['G_loss'] = []
            self.train_hist['per_epoch_time'] = []
            self.train_hist['total_time'] = []

            if self.gpu_mode:
                self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
            else:
                self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

            self.D.train()
            print('training start!!')
            start_time = time.time()
            for epoch in range(self.epoch):
                self.G.train()
                epoch_start_time = time.time()
                n_batch = 0.
                for iter in range(self.size_epoch):
                    n_batch += 1
                    x_ = sort_utils.get_batch(list_classes, classe, self.batch_size)

                    z_ = torch.rand((self.batch_size, self.z_dim))

                    if self.gpu_mode:
                        x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
                    else:
                        x_, z_ = Variable(x_), Variable(z_)

                    # update D network
                    self.D_optimizer.zero_grad()

                    D_real = self.D(x_)
                    D_real_err = torch.mean(torch.abs(D_real - x_))

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    D_fake_err = torch.mean(torch.abs(D_fake - G_))

                    D_loss = D_real_err - self.k * D_fake_err
                    self.train_hist['D_loss'].append(D_loss.data[0])

                    D_loss.backward()
                    self.D_optimizer.step()

                    # update G network
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    D_fake_err = torch.mean(torch.abs(D_fake - G_))

                    G_loss = D_fake_err
                    self.train_hist['G_loss'].append(G_loss.data[0])

                    G_loss.backward()
                    self.G_optimizer.step()

                    # convergence metric
                    temp_M = D_real_err + torch.abs(self.gamma * D_real_err - D_fake_err)

                    # operation for updating k
                    temp_k = self.k + self.lambda_ * (self.gamma * D_real_err - D_fake_err)
                    temp_k = temp_k.data[0]

                    # self.k = temp_k.data[0]
                    self.k = min(max(temp_k, 0), 1)
                    self.M = temp_M.data[0]

                    if ((iter + 1) % 100) == 0:
                        print("classe : [%1d] Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, M: %.8f, k: %.8f" %
                              (classe, (epoch + 1), (iter + 1), self.size_epoch,
                               D_loss.data[0], G_loss.data[0], self.M, self.k))

                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                self.visualize_results((epoch+1), classe)

            self.save_G(classe)

            result_dir = self.result_dir + '/' + 'classe-' + str(classe)
            utils.generate_animation(result_dir + '/' + self.model_name, epoch + 1)
            utils.loss_plot(self.train_hist, result_dir, self.model_name)

            np.savetxt(
                os.path.join(result_dir, 'began_training_' + self.dataset + '.txt'),
                np.transpose([self.train_hist['G_loss']]))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

