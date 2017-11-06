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
        #if conditional:
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
                samples = self.G(self.sample_z_)

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
