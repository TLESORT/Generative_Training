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
from discriminator import Discriminator
from encoder import Encoder
from load_dataset import load_dataset


class GenerativeModel(object):
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
        self.num_examples = args.num_examples
        self.c_criterion = nn.NLLLoss()
        self.size_epoch = args.size_epoch
        self.BCELoss = nn.BCELoss()

        if self.conditional:
            self.model_name = 'C' + self.model_name
        self.device = args.device
        # networks init

        # load dataset
        data_loader = load_dataset(self.dataset, self.batch_size, self.num_examples)
        self.data_loader_train = data_loader[0]
        self.data_loader_valid = data_loader[1]

        # BEGAN parameters
        self.gamma = 0.75
        if self.model_name == "BEGAN":
            self.lambda_ = 0.001
        elif self.model_name == "WGAN_GP":
            self.lambda_ = 0.25
        self.k = 0.

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
        self.D = Discriminator(self.dataset, self.conditional, self.model_name)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.model_name == 'VAE' or self.model_name == 'CVAE':
            self.E = Encoder(self.z_dim, self.dataset, self.conditional)
            self.E_optimizer = optim.Adam(self.E.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
            if self.gpu_mode:
                self.E.cuda()
        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise
        if self.model_name == 'VAE' or self.model_name == 'CVAE':
            self.sample_z_ = Variable(torch.randn((self.batch_size, self.z_dim, 1, 1)), volatile=True)
        else:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim, 1, 1)), volatile=True)

        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

        print("Model      : ", self.model_name)
        print("Dataset    : ", self.dataset)
        print("Num Ex     : ", self.num_examples)
        print("batch size : ", self.batch_size)
        print("z size     : ", self.z_dim)

    def test(self, predict, labels):
        correct = 0
        pred = predict.data.max(1)[1]
        correct = pred.eq(labels.data).cpu().sum()
        return correct, len(labels.data)

    def visualize_results(self, epoch, classe=None, fix=True):
        self.G.eval()
        dir_path = self.result_dir
        if classe is not None:
            dir_path = self.result_dir + '/classe-' + str(classe)

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
            if self.model_name == 'VAE' or self.model_name == 'CVAE':
                sample_z_ = Variable(torch.randn((self.batch_size, self.z_dim, 1, 1)), volatile=True)
            else:
                sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim, 1, 1)), volatile=True)

            if self.gpu_mode:
                sample_z_ = sample_z_.cuda(self.device)

            if self.conditional:
                samples = self.G(sample_z_, y_onehot)
            else:
                samples = self.G(self.sample_z_)

        if self.gpu_mode:
            print(samples.cpu().data.numpy().shape)
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          dir_path + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def sample(self, batch_size, classe=None):
        self.G.eval()
        if self.conditional:
            if self.model_name == 'VAE' or self.model_name == 'CVAE':
                z_ = torch.randn(batch_size, self.z_dim, 1, 1)
            else:
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
            if self.model_name == 'VAE' or self.model_name == 'CVAE':
                z_ = torch.randn(self.batch_size, 1, self.z_dim, 1, 1)
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

        i = 0
        self.G0 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G0.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G1 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G1.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G2 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G2.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G3 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G3.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G4 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G4.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G5 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G5.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G6 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G6.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G7 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G7.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G8 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G8.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1
        self.G9 = Generator(self.z_dim, self.dataset, self.conditional, self.model_name).cuda(self.device)
        self.G9.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '-' + str(i) + '_G.pkl')))
        i += 1

    def load(self):

        self.G.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '_G.pkl')))
        if self.model_name == 'VAE' or self.model_name == 'CVAE':
            self.E.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '_E.pkl')))
        else:
            self.D.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '_D.pkl')))

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

    def save_G(self, classe):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        torch.save(self.G.state_dict(), os.path.join(self.save_dir, self.model_name + '-' + str(classe) + '_G.pkl'))

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        torch.save(self.G.state_dict(), os.path.join(self.save_dir, self.model_name + '_G.pkl'))
        if self.model_name == 'VAE' or self.model_name == 'CVAE':
            torch.save(self.E.state_dict(), os.path.join(self.save_dir, self.model_name + '_E.pkl'))
        else:
            torch.save(self.D.state_dict(), os.path.join(self.save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(self.save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)
