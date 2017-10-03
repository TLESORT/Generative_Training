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

cuda = torch.cuda.is_available()
batch_size=512
log_interval=100
epochs=1500
seed=1
latent_space = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
resume = 'checkpoint2.pth.tar'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--train_vae', dest='train_vae', action='store_true')
parser.add_argument('--train_classifier', dest='train_classifier', action='store_true')
args = parser.parse_args()

### Save model
def save_checkpoint(state, is_best, filename=resume):
    torch.save(state, filename)
    # if is_best:
    #	shutil.copyfile(filename, 'model_best.pth.tar')


### Load dataset
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/Tmp/bordesfl/', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/Tmp/bordesfl/', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


### Saves images
def print_samples(prediction, nepoch, batch_size, filename_dest):
    plt.figure()
    batch_size_sqrt = int(np.sqrt(batch_size))
    input_dim = prediction[0].shape[1]
    prediction = np.clip(prediction, 0, 1)
    pred = prediction.reshape((batch_size_sqrt, batch_size_sqrt, input_dim, input_dim))
    pred = pred.swapaxes(2, 1)
    pred = pred.reshape((batch_size_sqrt*input_dim, batch_size_sqrt*input_dim))
    fig, ax = plt.subplots(figsize=(batch_size_sqrt, batch_size_sqrt))
    ax.axis('off')
    ax.imshow(pred, cmap='Greys_r')
    ax.grid()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(filename_dest, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    plt.close()


### Define VAE Model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(794, 1200)
        self.fc21 = nn.Linear(1200, latent_space)
        self.fc22 = nn.Linear(1200, latent_space)
        self.fc3 = nn.Linear(latent_space+10, 1200)
        self.fc4 = nn.Linear(1200, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h1 = self.relu(self.fc1(inputs))
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
        h3 = self.relu(self.fc3(inputs))
        h4 = self.sigmoid(self.fc4(h3))
        return h4.view(-1, 1, 28, 28)

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 784), c)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, c), mu, logvar


### Define classifier
class Classif(nn.Module):
    def __init__(self):
        super(Classif, self).__init__()
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


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    KLD = torch.mean(KLD)
    KLD /= 784

    return BCE + KLD, KLD, BCE


# Get samples and label from CVAE
def samples_VAE(batch_idx):
    hidden_vector=torch.randn((batch_idx,latent_space))
    hidden_vector=Variable(hidden_vector.cuda())
    y = torch.LongTensor(batch_idx,1).random_() % 10
    y_onehot = torch.FloatTensor(batch_size, 10)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1.0)
    y_onehot = Variable(y_onehot.cuda())
    output=model.decode(hidden_vector, y_onehot)
    return output, y


# Training function for VAE
def train_VAE(epoch):
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
        # VAE
        recon_batch, mu, logvar = model(data, target)
        loss, loss_ll, loss_recon = loss_function(recon_batch, data, mu, logvar)
        loss.backward(retain_variables=True)
        train_loss += loss.data[0]
        loss_ll_train += loss_ll.data[0]
        loss_recon_train += loss_recon.data[0]
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}, Average LogLikelihood: {:.4f}, Average Recon loss {:.4f}'.format(
          epoch, train_loss / batch_idx, loss_ll_train / batch_idx, loss_recon_train / batch_idx))
    if epoch % 100 == 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
	}, is_best=True)

# Training function for the classifier
def train_classifier(epoch):
    size_epoch=100
    model_classif.train()
    train_loss = 0
    train_loss_classif = 0
    dataiter = iter(train_loader)

    for batch_idx in range(size_epoch):
        data, target = samples_VAE(batch_size)
	# data, target = dataiter.next()
        if cuda:
            data, target = data.cuda(), target.cuda()
	# data = Variable(data)
        target = Variable(target.squeeze())
        optimizer.zero_grad()
        classif = model_classif(data)
        loss_classif = F.nll_loss(classif, target)
        loss_classif.backward(retain_variables=True)
        train_loss_classif += loss_classif.data[0]
        optimizer.step()
    print('====> Epoch: {} Average loss classif: {:.4f}'.format(
          epoch, train_loss_classif / size_epoch))

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
        test_loss_classif  += F.nll_loss(classif, target, size_average=False).data[0] # sum up batch loss
        pred = classif.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss_classif , correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))


if args.train_vae:
	print "Training CVAE"
	# Declare CVAE model
	model = VAE()
	if cuda:
	    model.cuda()
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	for epoch in range(1, epochs + 1):
	    train_VAE(epoch)

	output, labels = samples_VAE(batch_size)
	img=output.data.cpu().numpy().reshape((batch_size,1,28,28))
	print_samples(img[0:144], 1, 144, 'samples.png')

if args.train_classifier:
	epochs = 500
	print "Training Classifier with CVAE samples"
        model = VAE()
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
	model_classif=Classif()
	odel_classif=model_classif.cuda()
	optimizer = optim.Adam(model_classif.parameters(), lr=1e-3)

	for epoch in range(1, epochs + 1):
	    train_classifier(epoch)
	    if epoch % 10 == 0:
		test(epoch)
