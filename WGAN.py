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

from generator import Generator
from load_dataset import load_dataset

LAMBDA = 10


class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset='mnist', conditional=False):
        super(discriminator, self).__init__()
        self.dataset = dataset
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1
            self.output_dim = 1
        elif dataset == 'cifar10':
            self.input_height = 32
            self.input_width = 32
            self.input_dim = 62
            if conditional:
                self.input_dim += 10
            self.output_dim = 1
        elif dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 3
            self.output_dim = 1

        shape = 128 * (self.input_height // 4) * (self.input_width // 4)
        # if conditional:
        #    shape += 10

        if dataset == 'cifar10':
            ndf = 64
            self.ndf = ndf
            self.conv = nn.Sequential(
                nn.Conv2d(3, ndf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # nn.Sigmoid()
            )
            shape_fc = 0
            if conditional:
                shape_fc += 10
            self.fc = nn.Sequential(
                nn.Linear(ndf * 8 * 4 * 4 + shape_fc, self.output_dim),
                nn.Sigmoid(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(self.input_dim, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
            )
            self.fc = nn.Sequential(
                nn.Linear(shape, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, self.output_dim),
                nn.Sigmoid(),
            )
            self.aux_linear = nn.Linear(shape, 10)
            self.softmax = nn.Softmax()
            utils.initialize_weights(self)

    def forward(self, input, c=None):
        if self.dataset == 'cifar10':
            x = self.conv(input)
            x = x.view(-1, 4 * 4 * self.ndf * 8)
        else:
            x = self.conv(input)
            x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        final = self.fc(x)
        if c is not None:
            c = self.aux_linear(x)
            c = self.softmax(c)
            return final, c
        else:
            return final


class WGAN(object):
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
        self.nb_batch = args.nb_batch
        self.num_examples = args.num_examples
        self.c_criterion = nn.NLLLoss()

        if self.conditional:
            self.model_name = 'C' + self.model_name
        self.device = args.device
        # networks init

        # load dataset
        data_loader = load_dataset(self.dataset, self.batch_size, self.num_examples)
        self.data_loader_train = data_loader[0]
        self.data_loader_valid = data_loader[1]

        if self.dataset == 'mnist':
            self.z_dim = 20
            self.input_size = 1
            self.size = 28
        elif self.dataset == 'fashion-mnist':
            self.z_dim = 20
            self.input_size = 1
            self.size = 28

        elif self.dataset == 'cifar10':
            self.input_size = 3
            self.size = 32
            self.imageSize = 32
            self.z_dim = 100

        elif self.dataset == 'celebA':
            self.z_dim = 100

        self.G = Generator(self.z_dim, self.dataset, self.conditional, self.model_name)
        self.D = discriminator(self.dataset, self.conditional)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise
        if self.gpu_mode:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim, 1, 1)).cuda(), volatile=True)
        else:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim, 1, 1)), volatile=True)

    def test(self, predict, labels):
        correct = 0
        pred = predict.data.max(1)[1]
        correct = pred.eq(labels.data).cpu().sum()
        return correct, len(labels.data)

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
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/num_examples_' +
                                 str(self.num_examples) + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name, 'num_examples_' +
                                                      str(self.num_examples)), self.model_name)

    def train(self):

        list_classes = sort_utils.get_list_batch(self.data_loader_train,
                                                 self.nb_batch)  # list filled all classe sorted by class
        list_classes_valid = sort_utils.get_list_batch(self.data_loader_valid,
                                                       self.nb_batch)  # list filled all classe sorted by class
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

                sum_loss_train = 0.
                n_batch = 0.
                # for iter in range(self.nb_batch):
                for iter, (x_, t_) in enumerate(self.data_loader_train):
                    # x_ = sort_utils.get_batch(list_classes, classe, self.batch_size)
                    # x_ = torch.FloatTensor(x_)
                    # Apply mask on the data to get the correct class
                    mask_idx = torch.nonzero(t_ == classe)
                    if mask_idx.dim() == 0:
                        continue
                    x_ = torch.index_select(x_, 0, mask_idx[:, 0])
                    z_ = torch.rand((self.batch_size, self.z_dim, 1, 1))
                    t_ = torch.index_select(t_, 0, mask_idx[:, 0])
                    x_ = Variable(x_)
                    z_ = Variable(z_)
                    if self.gpu_mode:
                        x_ = x_.cuda(self.device)
                        t_ = t_.cuda(self.device)
                        z_ = z_.cuda(self.device)
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
                        self.train_hist['G_loss'].append(G_loss.data[0])

                        G_loss.backward()
                        self.G_optimizer.step()

                        self.train_hist['D_loss'].append(D_loss.data[0])

                    if ((iter + 1) % 100) == 0:
                        print("classe : [%1d] Epoch: [%2d] [%4d/%4d] G_loss: %.8f, D_loss: %.8f" %
                              (classe, (epoch + 1), (iter + 1), self.nb_batch, G_loss.data[0], D_loss.data[0]))
                    n_batch += 1
                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                self.visualize_results((epoch + 1), classe)
            self.save_G(classe)
            result_dir = self.result_dir + '/' + self.dataset + '/' + self.model_name + '/num_examples_' + \
                         str(self.num_examples) + '/' + 'classe-' + str(classe)
            utils.generate_animation(result_dir + '/' + self.model_name, epoch + 1)
            utils.loss_plot(self.train_hist, result_dir, self.model_name)

            np.savetxt(
                os.path.join(result_dir, 'vae_training_' + self.dataset + '.txt'),
                np.transpose([self.train_hist['G_loss']]))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    def train_old(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        list_classes = sort_utils.get_list_batch(self.data_loader,
                                                 self.nb_batch)  # list filled all classes sorted by class

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(
                torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(
                torch.zeros(self.batch_size, 1))

        self.D.train()

        print('training start!!')
        start_time = time.time()
        for classe in range(10):
            for epoch in range(self.epoch):
                self.G.train()
                epoch_start_time = time.time()
                for iter in range(self.nb_batch):
                    x_ = sort_utils.get_batch(list_classes, classe, self.batch_size)
                    z_ = torch.rand((self.batch_size, self.z_dim, 1, 1))

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
                        self.train_hist['G_loss'].append(G_loss.data[0])

                        G_loss.backward()
                        self.G_optimizer.step()

                        self.train_hist['D_loss'].append(D_loss.data[0])

                    if ((iter + 1) % 100) == 0:
                        print("classe : [%1d] Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                              (classe, (epoch + 1), (iter + 1), self.nb_batch, D_loss.data[0], G_loss.data[0]))
                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                self.visualize_results((epoch + 1), classe)
                self.save_G(classe)

            result_dir = self.result_dir + '/' + self.dataset + '/' + self.model_name + '/num_examples_' + \
                         str(self.num_examples) + '/' + 'classe-' + str(classe)
            utils.generate_animation(result_dir + '/' + self.model_name, self.epoch)
            utils.loss_plot(self.train_hist, result_dir, self.model_name)

            np.savetxt(
                os.path.join(self.result_dir, self.dataset, self.model_name, 'num_examples_' +
                             str(self.num_examples), 'classe-' + str(classe), 'wgan_training_' + self.dataset + '.txt'),
                np.transpose([self.train_hist['D_loss'], self.train_hist['G_loss']]))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()

    def visualize_results(self, epoch, classe=None, fix=True):
        self.G.eval()
        dir_path = self.result_dir + '/' + self.dataset + '/' + self.model_name + '/num_examples_' \
                   + str(self.num_examples)
        if classe is not None:
            dir_path = self.result_dir + '/' + self.dataset + '/' + self.model_name + '/num_examples_' + \
                       str(self.num_examples) + '/classe-' + str(classe)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        if self.conditional:
            y = torch.LongTensor(self.batch_size, 1).random_() % 10
            y_onehot = torch.FloatTensor(self.batch_size, 10)
            y_onehot.zero_()
            y_onehot.scatter_(1, y, 1.0)
            y_onehot = Variable(y_onehot.cuda(self.device))
        else:
            y_onehot = None
        if fix:
            """ fixed noise """
            if self.conditional:
                samples = self.G(self.sample_z_, y_onehot)
            else:
                samples = self.G(self.sample_z_)
        else:
            """ random noise """
            if self.gpu_mode:
                sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim, 1, 1)).cuda(self.device), volatile=True)
            else:
                sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim, 1, 1)), volatile=True)
            if self.conditional:
                samples = self.G(sample_z_, y_onehot)
            else:
                samples = self.G(self.sample_z)
            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          dir_path + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def sample(self, batch_size, classe=None):
        self.G.eval()
        if self.conditional:
            z_ = torch.rand(batch_size, self.z_dim, 1, 1)
            if self.gpu_mode:
                z_ = z_.cuda(self.device)
            if classe is not None:
                y = torch.ones(batch_size, 1) * classe
            else:
                y = torch.LongTensor(batch_size, 1).random_() % 10
            y_onehot = torch.FloatTensor(batch_size, 10)
            y_onehot.zero_()
            y_onehot.scatter_(1, y, 1.0)
            y_onehot = Variable(y_onehot.cuda(self.device))
            output = self.G(Variable(z_), y_onehot).data
        else:
            z_ = torch.rand(self.batch_size, 1, self.z_dim, 1, 1)
            if self.gpu_mode:
                z_ = z_.cuda(self.device)
            y = (torch.randperm(1000) % 10)[:batch_size]
            output = torch.FloatTensor(batch_size, self.input_size, self.size, self.size)
            if classe is not None:
                output = self.generators[classe](Variable(z_))
            else:
                for i in range(batch_size):
                    classe = int(y[i])
                    # output[i] = self.generators[classe](Variable(z_[i])).data.cpu()
                    G = self.get_generator(classe)
                    output[i] = G(Variable(z_[i])).data.cpu()
                if self.gpu_mode:
                    output = output.cuda(self.device)
        return output, y

    def save_G(self, classe):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name,
                                'num_examples_' + str(self.num_examples))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '-' + str(classe) + '_G.pkl'))

    def get_generator(self, nb):
        i = 0
        if nb == i:
            return self.G0.eval()
        i += 1
        if nb == i:
            return self.G1.eval()
        i += 1
        if nb == i:
            return self.G2.eval()
        i += 1
        if nb == i:
            return self.G3.eval()
        i += 1
        if nb == i:
            return self.G4.eval()
        i += 1
        if nb == i:
            return self.G5.eval()
        i += 1
        if nb == i:
            return self.G6.eval()
        i += 1
        if nb == i:
            return self.G7.eval()
        i += 1
        if nb == i:
            return self.G8.eval()
        i += 1
        if nb == i:
            return self.G9.eval()

    def load_generators(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name,
                                'num_examples_' + str(self.num_examples))
        paths = [x for x in os.listdir(save_dir) if x.endswith("_G.pkl")]
        paths.sort()

        '''
        self.generators = []

        for i in range(10):
        model_path = os.path.join(save_dir, paths[i])
        G=Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        if self.gpu_mode:
            self.generators.append(G)
        else:
            self.generators.append(self.G)

        '''
        i = 0
        self.G0 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G0.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G1 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G1.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G2 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G2.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G3 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G3.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G4 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G4.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G5 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G5.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G6 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G6.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G7 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G7.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G8 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G8.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G9 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G9.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name, 'num_examples_' + str(self.num_examples))

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

    '''
    def load_generators(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name, 'num_examples_' + str(self.num_examples))
        paths = [x for x in os.listdir(save_dir) if x.endswith("_G.pkl")]
        paths.sort()
        for i in range(10):
            model_path = os.path.join(save_dir, paths[i])
            self.G.load_state_dict(torch.load(model_path))
            self.generators.append(copy.deepcopy(self.G.cuda()))
    '''
