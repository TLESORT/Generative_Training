import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fashion import fashion
import utils
import sort_utils
import numpy as np

class Cifar10_Classifier(nn.Module):
    def __init__(self):
        super(Cifar10_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

class Fashion_Classifier(nn.Module):
    def __init__(self):
        super(Fashion_Classifier, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(32*4*4, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        return F.log_softmax(out)


class Mnist_Classifier(nn.Module):
    def __init__(self):
        super(Mnist_Classifier, self).__init__()
        self.input_height = 28
        self.input_width = 28
        self.input_dim = 1
        self.output_dim = 1
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class Trainer(object):
    def __init__(self, model, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 16
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.lr=args.lrC
        self.momentum=args.momentum
        self.log_interval=100
        self.size_epoch=1000
        self.trainer=args.trainer
        self.generator = model
        self.conditional = args.conditional
        # Load the generator parameters
        if self.conditional:
            self.generator.load()
        else:
            self.generators=self.generator.load_generators()

        # generators features
        if self.trainer=='GAN':
            self.z_dim = 62

        # load dataset
        if self.dataset == 'mnist':
            self.train_loader = DataLoader(datasets.MNIST('data/mnist', train=True, download=True,
                                                                          transform=transforms.Compose(
                                                                              [transforms.ToTensor()])),
                                                           batch_size=self.batch_size, shuffle=True)
            self.test_loader = DataLoader(datasets.MNIST('data/mnist', train=False, download=True,
                                                                          transform=transforms.Compose(
                                                                              [transforms.ToTensor()])),
                                                           batch_size=self.batch_size, shuffle=True)
        elif self.dataset == 'fashion-mnist':
            kwargs = {'num_workers': 1, 'pin_memory': True} if self.gpu_mode else {}

            self.train_loader = data.DataLoader(fashion('fashion_data', train=True, download=True, transform=transforms.ToTensor()),
                                 batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
            self.test_loader = data.DataLoader(fashion('fashion_data', train=False, download=True, transform=transforms.ToTensor()),
                                 batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

        elif self.dataset == 'celebA':
            self.data_loader = utils.load_celebA('data/celebA', transform=transforms.Compose(
                [transforms.CenterCrop(160), transforms.Scale(64), transforms.ToTensor()]), batch_size=self.batch_size,
                                                 shuffle=True)
        elif self.dataset == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform)
            self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                      shuffle=True, num_workers=2)

            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=transform)
            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=4,
            shuffle=False, num_workers=2)



        if self.dataset=='mnist':
            self.Classifier=Mnist_Classifier()
        elif self.dataset=='fashion-mnist':
            self.Classifier=Fashion_Classifier()
        elif self.dataset=='cifar10':
            self.Classifier=Cifar10_Classifier()

        self.optimizer = optim.SGD(self.Classifier.parameters(), lr=self.lr, momentum=self.momentum)
        if self.gpu_mode:
            self.Classifier=self.Classifier.cuda()

    def train_sort(self):
        print("Classic Training")
        self.Classifier.train()
        for epoch in range(1, self.epoch + 1):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if self.gpu_mode:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                self.optimizer.zero_grad()
                output = self.Classifier(data)
                loss = F.nll_loss(output, target)
                #print(loss)
                loss.backward()
                self.optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.data[0]))
            self.test()


    ########################################### Condtional Training functions ###########################################
    # Training function for the classifier
    def train_classifier(self, epoch):
        size_epoch=100
        self.Classifier.train()
        train_loss = 0
        train_loss_classif = 0
        # dataiter = iter(train_loader)
        correct = 0
        for batch_idx in range(size_epoch):
            data, target = self.generator.sample(self.batch_size)
            # data, target = dataiter.next()
            if self.gpu_mode:
                data, target = data.cuda(), target.cuda()
            # data = Variable(data)
            target = Variable(target.squeeze())
            self.optimizer.zero_grad()
            classif = self.Classifier(data)
            loss_classif = F.nll_loss(classif, target)
            loss_classif.backward(retain_variables=True)
            self.optimizer.step()
            train_loss_classif += loss_classif.data[0]
            pred = classif.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
        train_loss_classif /= np.float(size_epoch * self.batch_size)
        print('====> Epoch: {} Average loss classif: {:.4f}'.format(
            epoch, train_loss_classif))
        if epoch % 10 == 0:
            print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                train_loss_classif , correct, size_epoch * self.batch_size, 100. * correct / (size_epoch * self.batch_size)))
        return train_loss_classif, (correct / np.float(size_epoch * self.batch_size))


    # Test function for the classifier
    def test_classifier(self, epoch):
        self.Classifier.eval()
        test_loss = 0
        test_loss_classif = 0
        correct = 0
        for data, target in self.test_loader:
            if self.gpu_mode:
                data = data.cuda()
                target = target.cuda()
            data = Variable(data, volatile=True)
            target = Variable(target, volatile=True)
            classif = self.Classifier(data)
            test_loss_classif  += F.nll_loss(classif, target, size_average=False).data[0] # sum up batch loss
            pred = classif.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        test_loss_classif /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss_classif))
        if epoch % 10 == 0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss_classif , correct, len(self.test_loader.dataset), correct / 100.))
        return test_loss_classif, np.float(correct) / len(self.test_loader.dataset)


    def train_with_conditional_gen(self):
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        for epoch in range(1, self.epoch + 1):
            loss, acc = self.train_classifier(epoch)
            train_loss.append(loss)
            train_acc.append(acc)
            loss, acc = self.test_classifier(epoch)
            test_loss.append(loss)
            test_acc.append(acc)
        np.savetxt('gan_data_classif_'+self.dataset+'.txt', np.transpose([train_loss, train_acc, test_loss, test_acc]))

    ##################################### DEV ##############################################################
    def train_with_generator(self):
        #path="/home/timothee/PhD/VAE_Will_Train_U/Generative_Training/models/mnist/GAN"
        #path = os.path.join(self.save_dir, self.dataset, self.model_name)
        #generators = sort_utils.get_generators(path, model.G)
        print("Generators train me")
        for epoch in range(1, self.epoch + 1):
            for batch_idx in range(self.size_epoch):
                z_ = torch.rand((self.batch_size,1, self.z_dim))
                #if self.gpu_mode:
                #    z_= Variable(z_.cuda())
                #else:
                #    z_ = Variable(z_)
                data, target = self.get_generators_batch(z_)
                if self.gpu_mode:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                self.optimizer.zero_grad()
                output = self.Classifier(data)
                loss = F.nll_loss(output, target)
                #print(loss)
                loss.backward()
                self.optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.data[0]))
            self.test()

    #########################################################################################################
    def test(self):
        self.Classifier.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            if self.gpu_mode:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.Classifier(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
        100. * correct / len(self.test_loader.dataset)))


    def visualize_results(self, epoch, fix=True):
        print("visualize_results is not yet implemented for Classifier")

    def get_generators_batch(self, noise):
        gene_indice = (torch.randperm(1000) % 10)[:self.batch_size]
        batch = torch.FloatTensor(self.batch_size, 1, 28, 28)
        target = torch.LongTensor(self.batch_size)
        for i in range(self.batch_size):
            target[i] = int(gene_indice[i])
            gene = self.generators[target[i]]
            h = Variable(noise[i].cuda())
            batch[i] = gene(h).data.cpu()
        return batch, target
