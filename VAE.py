from __future__ import print_function
import utils
import time
import os
import pickle
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from Generative_Model import GenerativeModel
from encoder import Encoder
from load_dataset import get_iter_dataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader

class VAE(GenerativeModel):
    def __init__(self, args):
        super(VAE, self).__init__(args)
        if self.dataset == 'mnist' or self.dataset == 'fashion-mnist':
            self.z_dim = 20

        self.E = Encoder(self.z_dim, self.dataset, self.conditional)
        self.E_optimizer = optim.Adam(self.E.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        self.lr = args.lrD
        if self.gpu_mode:
            self.E.cuda(self.device)

        self.sample_z_ = Variable(torch.randn((self.batch_size, self.z_dim, 1, 1)), volatile=True)
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda(self.device)

        print("create G and D")
        self.G = Generator(self.z_dim, self.dataset, self.conditional, self.model_name)
        self.D = Discriminator(self.dataset, self.conditional, self.model_name)

        print("create G and D 's optimizers")
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda(self.device)
            self.D.cuda(self.device)

    def load(self):
        self.G.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '_G.pkl')))
        self.E.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '_E.pkl')))

    # save a generator, encoder and discriminator in a given class
    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        torch.save(self.G.state_dict(), os.path.join(self.save_dir, self.model_name + '_G.pkl'))
        torch.save(self.E.state_dict(), os.path.join(self.save_dir, self.model_name + '_E.pkl'))

        with open(os.path.join(self.save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def random_tensor(self, batch_size, z_dim):
        # From Normal distribution for VAE and CVAE
        return torch.randn((batch_size, z_dim, 1, 1))


    def train_all_classes(self):
        self.train_hist = {}
        self.train_hist['Train_loss'] = []
        self.train_hist['Valid_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.size_epoch = 1

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda(self.device)), Variable(
                torch.zeros(self.batch_size, 1).cuda(self.device))
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(
                torch.zeros(self.batch_size, 1))

        self.E.train()
        print('training start!!')
        start_time = time.time()
        best = 1000000

        self.data_loader_train  = DataLoader(self.dataset_train, batch_size=self.batch_size)
        self.data_loader_valid = DataLoader(self.dataset_valid, batch_size=self.batch_size)

        early_stop = 0
        for epoch in range(self.epoch):
            self.E.train()
            self.G.train()
            epoch_start_time = time.time()
            sum_loss_train = 0.
            n_batch = 0.
            for iter, (x_, t_) in enumerate(self.data_loader_train):
                y_onehot = torch.FloatTensor(t_.shape[0], 10)
                y_onehot.zero_()
                y_onehot.scatter_(1, t_[:, np.newaxis], 1.0)
                if self.gpu_mode:
                    x_ = Variable(x_.cuda(self.device))
                    if self.conditional:
                        y_onehot = Variable(y_onehot.cuda(self.device))
                else:
                    x_ = Variable(x_)
                self.E_optimizer.zero_grad()
                self.G_optimizer.zero_grad()
                # VAE
                z_, mu, logvar = self.E(x_, y_onehot)
                recon_batch = self.G(z_, y_onehot)

                G_loss = self.loss_function(recon_batch, x_, mu, logvar)
                sum_loss_train += G_loss.data[0]
                G_loss.backward() #retain_variables=True)

                self.E_optimizer.step()
                self.G_optimizer.step()
                n_batch += 1
                self.train_hist['Train_loss'].append(G_loss.data[0])

            sum_loss_train = sum_loss_train / np.float(n_batch)
            sum_loss_valid = 0.
            n_batch = 0.
            self.E.eval()
            self.G.eval()
            for iter, (x_, t_) in enumerate(self.data_loader_valid):
                n_batch += 1
                y_onehot = torch.FloatTensor(t_.shape[0], 10)
                y_onehot.zero_()
                y_onehot.scatter_(1, t_[:, np.newaxis], 1.0)
                if self.gpu_mode:
                    x_ = Variable(x_.cuda(self.device))
                    if self.conditional:
                        y_onehot = Variable(y_onehot.cuda(self.device))
                else:
                    x_ = Variable(x_)
                # VAE
                z_, mu, logvar = self.E(x_, y_onehot)
                recon_batch = self.G(z_, y_onehot)
                G_loss = self.loss_function(recon_batch, x_, mu, logvar)
                sum_loss_valid += G_loss.data[0]
                self.train_hist['Valid_loss'].append(G_loss.data[0])
            sum_loss_valid = sum_loss_valid / np.float(n_batch)
            print("Epoch: [%2d] Train_loss: %.8f, Valid_loss: %.8f" % ((epoch + 1), sum_loss_train, sum_loss_valid))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.visualize_results((epoch + 1))
            if sum_loss_valid < best:
                best = sum_loss_valid
                self.save()
                early_stop = 0.
            # We dit early stopping of the valid performance doesn't
            # improve anymore after 50 epochs
            if early_stop == 50:
                #break
                print("I should stop")
            else:
                early_stop += 1

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")
        utils.generate_animation(self.result_dir + '/' + self.model_name, self.epoch)
        # utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name,
        #                                               'num_examples_' + str(self.num_examples)), self.model_name)

        np.savetxt(
            os.path.join(self.result_dir + '/cvae_training_' +
                         self.dataset + '.txt'), np.transpose([self.train_hist['Train_loss']]))

    def train(self):

        #list_classes = sort_utils.get_list_batch(self.data_loader_train)  # list filled all classe sorted by class
        #list_classes_valid = sort_utils.get_list_batch(self.data_loader_valid)  # list filled all classe sorted by class
        print(' training start!! (no conditional)')
        start_time = time.time()

        for classe in range(10):
            self.train_hist = {}
            self.train_hist['D_loss'] = []
            self.train_hist['G_loss'] = []
            self.train_hist['per_epoch_time'] = []
            self.train_hist['total_time'] = []
            self.G.apply(self.G.weights_init)
            del self.E
            self.E = Encoder(self.z_dim, self.dataset, self.conditional)
            self.E_optimizer = optim.Adam(self.E.parameters(), lr=self.lr) #, lr=args.lrD, betas=(args.beta1, args.beta2))
            if self.gpu_mode:
                self.E.cuda(self.device)

            best = 100000
            self.data_loader_train = get_iter_dataset(self.dataset_train, self.list_class_train, self.batch_size,
                                                      classe)
            self.data_loader_valid = get_iter_dataset(self.dataset_valid, self.list_class_valid, self.batch_size,
                                                      classe)
            early_stop = 0.
            for epoch in range(self.epoch):

                epoch_start_time = time.time()
                # print("number of batch data")
                # print(len(self.data_loader_train))
                self.E.train()
                self.G.train()
                sum_loss_train = 0.
                n_batch = 0.
                #for iter in range(self.size_epoch):
                for iter, (x_, t_) in enumerate(self.data_loader_train):
                    n_batch += 1
                    #x_ = sort_utils.get_batch(list_classes, classe, self.batch_size)
                    #x_ = torch.FloatTensor(x_)
                    x_ = Variable(x_)
                    if self.gpu_mode:
                        x_ = x_.cuda(self.device)
                    # VAE
                    z_, mu, logvar = self.E(x_)
                    recon_batch = self.G(z_)

                    # train
                    self.G_optimizer.zero_grad()
                    self.E_optimizer.zero_grad()
                    g_loss = self.loss_function(recon_batch, x_, mu, logvar)
                    g_loss.backward()#retain_variables=True)
                    sum_loss_train += g_loss.data[0]
                    self.G_optimizer.step()
                    self.E_optimizer.step()

                    self.train_hist['D_loss'].append(g_loss.data[0])
                    self.train_hist['G_loss'].append(g_loss.data[0])

                    if ((iter + 1) % 100) == 0:
                        print("classe : [%1d] Epoch: [%2d] [%4d/%4d] G_loss: %.8f, E_loss: %.8f" %
                              (classe, (epoch + 1), (iter + 1), self.size_epoch, g_loss.data[0], g_loss.data[0]))
                sum_loss_train = sum_loss_train / np.float(n_batch)
                sum_loss_valid = 0.
                n_batch = 0.
                n_batch = 1.
                self.E.eval()
                self.G.eval()
                for iter, (x_, t_) in enumerate(self.data_loader_valid):
                    n_batch += 1
                    max_val, max_indice = torch.max(t_, 0)
                    mask_idx = torch.nonzero(t_ == classe)
                    if mask_idx.dim() == 0:
                        continue
                    x_ = torch.index_select(x_, 0, mask_idx[:, 0])
                    t_ = torch.index_select(t_, 0, mask_idx[:, 0])
                    if self.gpu_mode:
                        x_ = Variable(x_.cuda(self.device), volatile=True)
                    else:
                        x_ = Variable(x_)
                    # VAE
                    z_, mu, logvar = self.E(x_)
                    recon_batch = self.G(z_)

                    G_loss = self.loss_function(recon_batch, x_, mu, logvar)
                    sum_loss_valid += G_loss.data[0]

                sum_loss_valid = sum_loss_valid / np.float(n_batch)
                print("classe : [%1d] Epoch: [%2d] Train_loss: %.8f, Valid_loss: %.8f" % (
                classe, (epoch + 1), sum_loss_train, sum_loss_valid))
                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                self.visualize_results((epoch + 1), classe)
                if sum_loss_valid < best:
                    best = sum_loss_valid
                    self.save_G(classe)
                    early_stop = 0.
                # We dit early stopping of the valid performance doesn't
                # improve anymore after 50 epochs
                if early_stop == 50:
                    break
                else:
                    early_stop += 1
            result_dir = self.result_dir + '/' + 'classe-' + str(classe)
            utils.generate_animation(result_dir + '/' + self.model_name, epoch + 1)
            utils.loss_plot(self.train_hist, result_dir, self.model_name)

            np.savetxt(
                os.path.join(result_dir, 'vae_training_' + self.dataset + '.txt'),
                np.transpose([self.train_hist['G_loss']]))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    def loss_function(self, recon_x, x, mu, logvar):
        # BCE = F.binary_cross_entropy(recon_x, x).cuda(self.device)
        if self.dataset == 'mnist' or self.dataset == 'fashion-mnist':
            reconstruction_function = nn.BCELoss()
        else:
            reconstruction_function = nn.MSELoss()
        # reconstruction_function.size_average = False
        BCE = reconstruction_function(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        #KLD = torch.mean(KLD)
        #KLD /= 784

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        if self.gpu_mode:
            BCE = BCE.cuda(self.device)
            KLD = KLD.cuda(self.device)
        return BCE + KLD
