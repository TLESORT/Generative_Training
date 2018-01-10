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
import copy


from load_dataset import load_dataset
from Generative_Model import GenerativeModel

LAMBDA = 10



class WGAN(GenerativeModel):

    def calc_gradient_penalty(self, real_data, fake_data, batch_size):
        # print real_data.size()
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()  # if use_cuda else alpha
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def train_all_classes(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(
                torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(
                torch.zeros(self.batch_size, 1))

        self.D.train()
        print('training start!!')
        start_time = time.time()
        one = torch.FloatTensor([1])
        mone = one * -1
        one = one.cuda()
        mone = mone.cuda()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, t_) in enumerate(self.data_loader_train):
                z_ = torch.rand(((x_.size(0)), self.z_dim))
                if self.conditional:
                    y_onehot = torch.FloatTensor(t_.shape[0], 10)
                    y_onehot.zero_()
                    y_onehot.scatter_(1, t_[:, np.newaxis], 1.0)
                else:
                    y_onehot = None
                if self.gpu_mode:
                    x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
                    if self.conditional:
                        y_onehot = Variable(y_onehot.cuda(self.device))
                else:
                    x_, z_ = Variable(x_), Variable(z_)

                t_ = Variable(t_.cuda())
                # update D network
                self.D_optimizer.zero_grad()

                D_real, Real_softmax = self.D(x_, y_onehot)
                D_real_loss = torch.mean(D_real)  # + self.c_criterion(Real_softmax, t_)
                D_real_loss.backward(mone)
                correct, length = self.test(Real_softmax, t_)

                G_ = self.G(z_, y_onehot)
                D_fake, fake_softmax = self.D(G_, y_onehot)
                D_fake_loss = torch.mean(D_fake)  # + self.c_criterion(fake_softmax, t_)
                D_fake_loss.backward(one)
                gradient_penalty = self.calc_gradient_penalty(x_.data, G_.data, x_.size(0))
                gradient_penalty.backward()
                D_loss = D_fake_loss - D_real_loss + gradient_penalty

                self.D_optimizer.step()

                # clipping D
                # for p in self.D.parameters():
                #    p.data.clamp_(-self.c, self.c)
                # train with gradient penalty
                # gradient_penalty = self.calc_gradient_penalty(x_.data, G_.data)
                # gradient_penalty.backward()

                if ((iter + 1) % self.n_critic) == 0:
                    # update G network
                    for p in self.D.parameters():
                        p.requires_grad = False  # to avoid computation
                        self.G.zero_grad()
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_, y_onehot)
                    D_fake, Real_softmax = self.D(G_, y_onehot)
                    G_loss = torch.mean(D_fake)  # + self.c_criterion(Real_softmax, t_)
                    self.train_hist['G_loss'].append(G_loss.data[0])

                    G_loss.backward(mone)
                    self.G_optimizer.step()

                    for p in self.D.parameters():
                        p.requires_grad = True  # to restore computation

                    self.train_hist['D_loss'].append(D_loss.data[0])

            print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f Accuracy: %.4f / %.4f = %.4f" %
                  ((epoch + 1), (iter + 1), self.data_loader_train.dataset.__len__() // self.batch_size,
                   D_loss.data[0], G_loss.data[0], correct, length, 100. * correct / length))

            if epoch % 2 == 0:
                self.visualize_results((epoch + 1))
                self.save()

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            # self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()

        utils.generate_animation(self.result_dir + '/' + self.model_name, self.epoch)
        utils.loss_plot(self.train_hist, self.save_dir, self.model_name)

    def train(self):

        list_classes = sort_utils.get_list_batch(self.data_loader_train)  # list filled all classe sorted by class
        #list_classes_valid = sort_utils.get_list_batch(self.data_loader_valid)  # list filled all classe sorted by class


        self.pretrain()

        print(' training start!! (no conditional)')
        start_time = time.time()
        for classe in range(10):
            self.train_hist = {}
            self.train_hist['D_loss'] = []
            self.train_hist['G_loss'] = []
            self.train_hist['per_epoch_time'] = []
            self.train_hist['total_time'] = []
            self.G.train()
            best = 100000

            for epoch in range(self.epoch):

                epoch_start_time = time.time()
                n_batch = 0.
                for iter in range(self.size_epoch):
                    n_batch += 1
                    x_ = sort_utils.get_batch(list_classes, classe, self.batch_size)

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

    def pretrain(self, epoch_pretrain=1):

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(
                torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(
                torch.zeros(self.batch_size, 1))

        self.D.train()
        print('pretraining start!!')
        for epoch in range(epoch_pretrain):
            self.G.train()
            for iter, (x_, _) in enumerate(self.data_loader_train):
                if iter == self.data_loader_train.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))

                if self.gpu_mode:
                    x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
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

