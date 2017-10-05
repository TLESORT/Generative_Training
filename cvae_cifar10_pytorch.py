import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import shutil
import argparse
import os

root = '/Tmp/bordesfl/'
cuda = torch.cuda.is_available()
batch_size = 512
log_interval = 100
epochs = 500
seed = 1
latent_space = 100
input_size = 3 * 32 * 32
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
resume = 'checkpoint_cifar10.pth.tar'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--train_vae', dest='train_vae', action='store_true')
parser.add_argument('--train_classifier', dest='train_classifier', action='store_true')
args = parser.parse_args()


### Save model
def save_checkpoint(state, is_best, filename=resume):
    torch.save(state, filename)
    # if is_best:
    #	shutil.copyfile(filename, 'model_best.pth.tar')


### Load dataset CIFAR10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root=root, train=True,
                            download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, num_workers=8)

testset = datasets.CIFAR10(root=root, train=False,
                           download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)


### Saves images
def print_samples(prediction, nepoch, batch_size, filename_dest):
    plt.figure()
    batch_size_sqrt = int(np.sqrt(batch_size))
    input_dim = prediction[0].shape[1]
    prediction = np.clip(prediction, 0, 1)
    input_channel = prediction[0].shape[0]
    pred = np.rollaxis(prediction.reshape((batch_size_sqrt, batch_size_sqrt, input_channel, input_dim, input_dim)), 2,
                       5)
    pred = pred.swapaxes(2, 1)
    pred = pred.reshape((batch_size_sqrt * input_dim, batch_size_sqrt * input_dim, input_channel))
    fig, ax = plt.subplots(figsize=(batch_size_sqrt, batch_size_sqrt))
    ax.axis('off')
    ax.imshow(pred)
    ax.grid()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(filename_dest, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    plt.close()


### Define CVAE Model
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)  # 28
        self.pool = nn.MaxPool2d(2, 2)  # 14
        self.conv2 = nn.Conv2d(64, 64, 5)  # 10
        self.conv3 = nn.Conv2d(64, 64, 3)  # 8
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2)  # 3
        self.conv5 = nn.Conv2d(128, 128, 3)  # 1
        self.conv6 = nn.Conv2d(128, 64, 1)
        # self.conv61 = nn.Conv2d(64, 64, 1)
        self.conv7_mu = nn.Conv2d(64, latent_space, 1)
        self.conv7_var = nn.Conv2d(64, latent_space, 1)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.fc = nn.Linear(74, 400)
        self.fc21 = nn.Linear(400, latent_space)
        self.fc22 = nn.Linear(400, latent_space)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(latent_space + 10, 128, 3, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 3, 2, 0)
        self.deconv2_bn = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 3, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 64, 3, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 5, 1, 0)
        self.deconv5_bn = nn.BatchNorm2d(32)
        self.deconv6 = nn.ConvTranspose2d(32, 3, 4, 1, 0)
        self.tanh = nn.Tanh()

    def encode(self, x, c):
        c1 = self.relu(self.conv1(x))
        p1 = self.pool(c1)
        c2 = self.relu(self.conv2(p1))
        c3 = self.relu(self.conv3(c2))
        c4 = self.relu(self.conv4(c3))
        c5 = self.relu(self.conv5(c4))
        c6 = self.relu(self.conv6(c5))
        # c61 = self.relu(self.conv61(c6))
        h1 = self.fc(torch.cat([c6.view(-1, 64), c], 1))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        x = F.relu(self.deconv1_bn(self.deconv1(inputs.view(-1, latent_space + 10, 1, 1))))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        x = F.sigmoid(self.deconv6(x))
        return x.view(-1, 3, 32, 32)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, c), mu, logvar


### Define classifier
class Classif(nn.Module):
    def __init__(self):
        super(Classif, self).__init__()
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


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_size))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    KLD = torch.mean(KLD)
    KLD /= input_size

    return BCE + KLD, KLD, BCE


# Get samples and label from CVAE
def samples_CVAE(batch_idx):
    hidden_vector = torch.randn((batch_idx, latent_space))
    hidden_vector = Variable(hidden_vector.cuda())
    y = torch.LongTensor(batch_idx, 1).random_() % 10
    y_onehot = torch.FloatTensor(batch_size, 10)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1.0)
    y_onehot = Variable(y_onehot.cuda())
    output = model.decode(hidden_vector, y_onehot)
    return output, y


# Training function for CVAE
def train_CVAE(epoch):
    model.train()
    train_loss = 0
    loss_ll_train = 0
    loss_recon_train = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        y_onehot = torch.FloatTensor(target.shape[0], 10)
        y_onehot.zero_()
        y_onehot.scatter_(1, target[:, np.newaxis], 1.0)
        if cuda:
            data, y_onehot = data.cuda(), y_onehot.cuda()
        data, target = Variable(data), Variable(y_onehot)
        optimizer.zero_grad()
        # CVAE
        recon_batch, mu, logvar = model(data, target)
        loss, loss_ll, loss_recon = loss_function(recon_batch, data, mu, logvar)
        loss.backward(retain_variables=True)
        train_loss += loss.data[0]
        loss_ll_train += loss_ll.data[0]
        loss_recon_train += loss_recon.data[0]
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}, Average LogLikelihood: {:.4f}, Average Recon loss {:.4f}'.format(
        epoch, train_loss / batch_idx, loss_ll_train / batch_idx, loss_recon_train / batch_idx))
    if epoch % 10 == 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=True)


# Training function for the classifier
def train_classifier(epoch):
    size_epoch = 100
    model_classif.train()
    train_loss = 0
    train_loss_classif = 0
    dataiter = iter(train_loader)
    correct = 0
    for batch_idx in range(size_epoch):
        data, target = samples_CVAE(batch_size)
        # data, target = dataiter.next()
        if cuda:
            data, target = data.cuda(), target.cuda()
        # data = Variable(data)
        target = Variable(target.squeeze())
        optimizer.zero_grad()
        classif = model_classif(data)
        loss_classif = F.nll_loss(classif, target)
        loss_classif.backward(retain_variables=True)
        optimizer.step()
        train_loss_classif += loss_classif.data[0]
        pred = classif.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    train_loss_classif /= np.float(size_epoch * batch_size)
    print('====> Epoch: {} Average loss classif: {:.4f}'.format(
        epoch, train_loss_classif))
    if epoch % 10 == 0:
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss_classif, correct, size_epoch * batch_size, 100. * correct / (size_epoch * batch_size)))
    return train_loss_classif, (correct / np.float(size_epoch * batch_size))


# Test function for the classifier
def test(epoch):
    model_classif.eval()
    test_loss = 0
    test_loss_classif = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data = data.cuda()
            target = target.cuda()
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)
        classif = model_classif(data)
        test_loss_classif += F.nll_loss(classif, target, size_average=False).data[0]  # sum up batch loss
        pred = classif.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader.dataset)
    test_loss_classif /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss_classif))
    if epoch % 10 == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss_classif, correct, len(test_loader.dataset), correct / 100.))
    return test_loss_classif, np.float(correct) / len(test_loader.dataset)


if args.train_vae:
    print "Training CVAE"
    # Declare CVAE model
    model = CVAE()
    if cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        train_CVAE(epoch)

    output, labels = samples_CVAE(batch_size)
    img = output.data.cpu().numpy().reshape((batch_size, 3, 32, 32))
    print_samples(img[0:144], 1, 144, 'samples_cifar10.png')

if args.train_classifier:
    epochs = 100
    print "Training Classifier with CVAE samples"
    model = CVAE()
    if cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    # Declare Classifier model
    model_classif = Classif()
    odel_classif = model_classif.cuda()
    optimizer = optim.Adam(model_classif.parameters(), lr=1e-3)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(1, epochs + 1):
        loss, acc = train_classifier(epoch)
        train_loss.append(loss)
        train_acc.append(acc)
        loss, acc = test(epoch)
        test_loss.append(loss)
        test_acc.append(acc)
    np.savetxt('data_classif_cifar10.txt', np.transpose([train_loss, train_acc, test_loss, test_acc]))
