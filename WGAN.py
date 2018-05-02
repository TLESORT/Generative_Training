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
from load_dataset import get_iter_dataset
import copy


from load_dataset import load_dataset
from Generative_Model import GenerativeModel

LAMBDA = 10



class WGAN(GenerativeModel):
    def __init__(self, args):
        super(WGAN, self).__init__(args)
        self.c = 0.01  # clipping value
        self.n_critic = 5  # the number of iterations of the critic per generator iteration

    def train(self):

        # list_classes = sort_utils.get_list_batch(self.data_loader_train)  # list filled all classe sorted by class
        #list_classes_valid = sort_utils.get_list_batch(self.data_loader_valid)  # list filled all classe sorted by class


        #self.pretrain()

        print(' training start!! (no conditional)')
        start_time = time.time()
        for classe in range(10):
            self.train_hist = {}
            self.train_hist['D_loss'] = []
            self.train_hist['G_loss'] = []
            self.train_hist['per_epoch_time'] = []
            self.train_hist['total_time'] = []
            self.G.apply(self.G.weights_init)
            self.G.train()
            best = 100000
            data_loader_train = get_iter_dataset(self.dataset_train, self.list_class_train, self.batch_size, classe)

            print("Classe: " + str(classe))
            for epoch in range(self.epoch):

                epoch_start_time = time.time()
                n_batch = 0.
                # for iter in range(self.size_epoch):
                for iter, (x_, t_) in enumerate(data_loader_train):
                    n_batch += 1
                    # x_ = sort_utils.get_batch(list_classes, classe, self.batch_size)

                    z_ = torch.rand((self.batch_size, self.z_dim, 1, 1))
                    if self.gpu_mode:
                        x_, z_ = Variable(x_.cuda(self.device)), Variable(z_.cuda(self.device))
                    else:
                        x_, z_ = Variable(x_), Variable(z_)
                    self.D_optimizer.zero_grad()
                    D_real = self.D(x_)
                    D_real_loss = -torch.mean(D_real)

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    D_fake_loss = torch.mean(D_fake)

                    D_loss = D_real_loss + D_fake_loss

                    D_loss.backward()
                    self.D_optimizer.step()

                    #print("FID :")
                    #print(self.compute_FID(G_, x_))

                    # clipping D
                    for p in self.D.parameters():
                        p.data.clamp_(-self.c, self.c)

                    if ((iter + 1) % self.n_critic) == 0:
                        # update G network
                        self.G_optimizer.zero_grad()

                        G_ = self.G(z_)
                        D_fake = self.D(G_)
                        G_loss = -torch.mean(D_fake)
                        self.train_hist['G_loss'].append(G_loss.data[0])

                        G_loss.backward()
                        self.G_optimizer.step()

                        self.train_hist['D_loss'].append(D_loss.data[0])

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

    def pretrain(self, epoch_pretrain=5):

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda(self.device)), Variable(
                torch.zeros(self.batch_size, 1).cuda(self.device))
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(
                torch.zeros(self.batch_size, 1))

        self.D.train()
        print('pretraining start!!')
        self.data_loader_train = get_iter_dataset(self.dataset_train, self.list_class_train)
        for nb in range(int(50000/self.num_examples)):
            for epoch in range(epoch_pretrain):
                self.G.train()
                for iter, (x_, _) in enumerate(self.data_loader_train):
                    if iter == self.data_loader_train.dataset.__len__() // self.batch_size:
                        break

                    z_ = torch.rand((self.batch_size, self.z_dim))

                    if self.gpu_mode:
                        x_, z_ = Variable(x_.cuda(self.device)), Variable(z_.cuda(self.device))
                    else:
                        x_, z_ = Variable(x_), Variable(z_)

                    # update D network
                    self.D_optimizer.zero_grad()

                    D_real = self.D(x_)
                    D_real_loss = -torch.mean(D_real)

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    D_fake_loss = torch.mean(D_fake)

                    D_loss = D_real_loss + D_fake_loss

                    D_loss.backward()
                    self.D_optimizer.step()

                    # clipping D
                    for p in self.D.parameters():
                        p.data.clamp_(-self.c, self.c)

                    if ((iter + 1) % self.n_critic) == 0:
                        # update G network
                        self.G_optimizer.zero_grad()

                        G_ = self.G(z_)
                        D_fake = self.D(G_)
                        G_loss = -torch.mean(D_fake)

                        G_loss.backward()
                        self.G_optimizer.step()
        print('pretraining end!!')

