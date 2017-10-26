import os
import torch
import torchvision
import copy
import pickle
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from fashion import fashion
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from load_dataset import load_dataset, load_dataset_test
import utils
import sort_utils
import numpy as np
import matplotlib as mpl
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

mpl.use('Agg')
import matplotlib.pyplot as plt


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


class Cifar10_Classifier(nn.Module):
    def __init__(self):
        super(Cifar10_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 1 * 1, 120)  # nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))  # self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)


class CelebA_Classifier(nn.Module):
    def __init__(self):
        super(CelebA_Classifier, self).__init__()
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
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

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
        self.sample_num = 64
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.sample_dir = args.sample_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.lr = args.lrC
        self.momentum = args.momentum
        self.log_interval = 100
        self.nb_batch = args.nb_batch
        self.gan_type = args.gan_type
        self.generator = model
        self.conditional = args.conditional
        self.device = args.device
        self.tau = args.tau
        self.num_examples = args.num_examples
        # Parameter for isotropic noise
        self.sigma = args.sigma
        self.tresh_masking_noise = args.tresh_masking_noise

        # Load the generator parameters
        if self.gan_type != "Classifier":
            if self.conditional:
                self.generator.load()
            else:
                self.generators = self.generator.load_generators()

        # load dataset
        data_loader = load_dataset(self.dataset, self.batch_size, self.num_examples)
        self.train_loader = data_loader[0]
        self.valid_loader = data_loader[1]
        self.test_loader = load_dataset_test(self.dataset, self.batch_size)

        if self.dataset == 'mnist':
            self.input_size = 1
            self.size = 28
            self.Classifier = Mnist_Classifier()
        elif self.dataset == 'fashion-mnist':
            self.input_size = 1
            self.size = 28
            self.Classifier = Fashion_Classifier()
        elif self.dataset == 'cifar10':
            self.input_size = 3
            self.size = 32
            self.Classifier = Cifar10_Classifier()
        elif self.dataset == 'celebA':
            self.Classifier = CelebA_Classifier()

        if self.gpu_mode:
            self.Classifier = self.Classifier.cuda(self.device)

        self.optimizer = optim.Adam(self.Classifier.parameters(), lr=self.lr, betas=(args.beta1, args.beta2))

    def knn(self):
        print("Training KNN Classifier")
        # Declare Classifier model
        data_samples = []
        label_samples = []

        # Training knn
        neigh = KNeighborsClassifier(n_neighbors=1)
        # We get the test data
        for i, (d, t) in enumerate(self.test_loader):
            if i == 0:
                data_test = d
                label_test = t
            else:
                data_test = torch.cat((data_test, d))
                label_test = torch.cat((label_test, t))
        data_test = data_test.numpy().reshape(-1, 784)
        label_test = label_test.numpy()
        # We get the training data
        for i, (d, t) in enumerate(self.train_loader):
            if i == 0:
                data_train = d
                label_train = t
            else:
                data_train = torch.cat((data_train, d))
                label_train = torch.cat((label_train, t))
        data_train = data_train.numpy().reshape(-1, 784)
        label_train = label_train.numpy()
        data_train[0:datas_train.size(0)*(1-self.tau)]
        label_train[0:labels_train.size(0)*(1-self.tau)]
        # We get samples from the models
        for i in range((labels_train.size(0)*self.tau)/self.batch_size):
            data, label = self.generator.sample(self.batch_size)
            data_samples.append(data.data.cpu().numpy())
            label_samples.append(label.cpu().numpy())
        # We concatenate training and gen samples
        data_samples = np.concatenate(data_samples).reshape(-1, 784)
        label_samples = np.concatenate(label_samples).squeeze()
        data = np.concatenate([data_train, data_samples])
        labels = np.concatenate([label_train, label_samples])
        # We train knn
        neigh.fit(data, labels)
        # We use it as prection
        predictions = neigh.predict(data_test)
        accuracy = np.sum(predictions == label_test) / np.float(data_label.shape[0])
        print(accuracy)
        np.savetxt(os.path.join(save_dir, 'best_score_knn_' + self.dataset + '.txt'),
                np.transpose([accuracy]))


    def train_classic(self):
        best_accuracy = 0
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        test_loss = []
        test_acc = []
        test_acc_classes = []

        # Training classifier
        for epoch in range(1, self.epoch + 1):
            tr_loss, tr_acc, v_loss, v_acc = self.train_classifier(epoch)
            train_loss.append(tr_loss)
            train_acc.append(tr_acc)
            val_loss.append(v_loss)
            val_acc.append(v_acc)
            # Save best model
            if v_acc > best_accuracy:
                best_accuracy = v_acc
                self.save(best=True)
                early_stop = 0.
            if early_stop == 60:
                break
            else:
                early_stop += 1
        # Then load best model
        self.load()
        if self.tresh_masking_noise > 0:
            name='tresh_'+str(self.tresh_masking_noise)
        elif self.sigma > 0:
            name='sigma_'+str(self.sigma)
        else:
            name='ref'
        loss, test_acc, test_acc_classes = self.test()  # self.test_classifier(epoch)
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name, 'num_examples_' + str(self.num_examples))
        np.savetxt(os.path.join(save_dir, 'data_classif_'+ name + self.dataset + '.txt'),
                   np.transpose([train_loss, train_acc, val_loss, val_acc]))
        np.savetxt(os.path.join(save_dir, 'best_score_classif_' + name + self.dataset + '.txt'),
                np.transpose([test_acc]))
        np.savetxt(os.path.join(save_dir, 'data_classif_classes' + name + self.dataset + '.txt'),
                   np.transpose([test_acc_classes]))


    def add_gen_batch2Training(self, batch_size):
        data, target = self.generator.sample(batch_size)

        if self.gpu_mode:
            data, target = data.cuda(self.device), target.cuda(self.device)
        batch = Variable(data)
        label = Variable(target.squeeze())
        self.optimizer.zero_grad()
        classif = self.Classifier(batch)
        loss_classif = F.nll_loss(classif, label)
        loss_classif.backward()
        self.optimizer.step()
        train_loss_classif = loss_classif.data[0]
        pred = classif.data.max(1)[1]  # get the index of the max log-probability
        correct = pred.eq(label.data).cpu().sum()

        return correct, train_loss_classif

    def train_classifier(self, epoch):
        self.Classifier.train()
        train_loss_classif = 0
        val_loss_classif = 0
        dataiter = iter(self.train_loader)
        best_accuracy = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            # batch_gen_size = int(self.tau * target_real.shape[0])
             #data, target = self.generator.sample(self.batch_size)
            # We take either traning data
            if torch.rand(1)[0] > self.tau: #NB : if tau < 0 their is no data augmentation
                if self.tau == 0:
                    if self.sigma > 0:
                        data = data + torch.zeros(data.size()).normal_(0, self.sigma)
                    if self.tresh_masking_noise > 0:
                        data = data * (torch.rand(data.shape) < self.tresh_masking_noise).type(torch.FloatTensor)
                if self.gpu_mode:
                    data, target = data.cuda(self.device), target.cuda(self.device)
                batch = Variable(data)
                label = Variable(target.squeeze())
                self.optimizer.zero_grad()
                classif = self.Classifier(batch)
                loss_classif = F.nll_loss(classif, label)
                loss_classif.backward()
                self.optimizer.step()
                train_loss_classif += loss_classif.data[0]
                pred = classif.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(label.data).cpu().sum()
            # or generated data
            else:
                corr, loss = self.add_gen_batch2Training(data.size(0))
                correct += corr
                train_loss_classif += loss

            """ I am not sure we need that
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx, self.nb_batch,
              self.nb_batcv,sv      100. * batch_idx / self.nb_batch))
            """
        train_loss_classif /= (np.float(self.num_examples))
        train_accuracy = 100. * correct / np.float(self.num_examples)

        correct = 0
        for batch_idx, (data, target) in enumerate(self.valid_loader):
            if self.gpu_mode:
                data, target = data.cuda(self.device), target.cuda(self.device)
            batch = Variable(data)
            label = Variable(target.squeeze())
            classif = self.Classifier(batch)
            loss_classif = F.nll_loss(classif, label)
            val_loss_classif += loss_classif.data[0]
            pred = classif.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(label.data).cpu().sum()

        val_loss_classif /= (np.float(len(self.valid_loader.sampler)))
        valid_accuracy = 100. * correct / np.float(len(self.valid_loader.sampler))

        print('Epoch: {} Train set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n Valid set: Average loss: {:.4f}, Accuracy: ({:.0f}%)'.format(
            epoch, train_loss_classif, train_accuracy, val_loss_classif, valid_accuracy))
        return train_loss_classif, train_accuracy, val_loss_classif, valid_accuracy

    def train_with_generator(self):
        best_accuracy = 0
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        test_loss = []
        test_acc = []
        test_acc_classes = []

        self.visualize_Samples()

        # Training classifier
        for epoch in range(1, self.epoch + 1):
            tr_loss, tr_acc, v_loss, v_acc = self.train_classifier(epoch)
            train_loss.append(tr_loss)
            train_acc.append(tr_acc)
            val_loss.append(v_loss)
            val_acc.append(v_acc)
            # Save best model
            if v_acc > best_accuracy:
                best_accuracy = v_acc
                self.save(best=True)
                #print(best_accuracy)
                early_stop = 0.
            if early_stop == 60:
                break
            else:
                early_stop += 1

            """ I think we could remove that
            if epoch % 1 == 0:
                loss, acc, acc_classes = self.test()  # self.test_classifier(epoch)
                test_loss.append(loss)
                test_acc.append(acc)
                test_acc_classes.append(acc_classes)
                if acc > best_accuracy:
                    best_accuracy = acc
                    self.save(best=True)
                    print(best_accuracy)
                #else:
                #     self.save()
                # self.compute_KLD() #we don't use it for instance and it takes some times...
            """
        # Then load best model
        self.load()
        loss, test_acc, test_acc_classes = self.test()  # self.test_classifier(epoch)
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name, 'num_examples_' + str(self.num_examples))
        np.savetxt(os.path.join(save_dir, 'data_classif_' + self.dataset + '-tau' + str(self.tau) + '.txt'),
                   np.transpose([train_loss, train_acc, val_loss, val_acc]))
        np.savetxt(os.path.join(save_dir, 'best_score_classif_' + self.dataset + '-tau' + str(self.tau) + '.txt'),
                np.transpose([test_acc]))
        np.savetxt(os.path.join(save_dir, 'data_classif_classes' + self.dataset + '-tau' + str(self.tau) + '.txt'),
                   np.transpose([test_acc_classes]))

    def test(self):
        self.Classifier.eval()
        test_loss = 0
        correct = 0
        classe_prediction = np.zeros(10)
        classe_total = np.zeros(10)
        classe_wrong = np.zeros(10)  # Images wrongly attributed to a particular class

        for data, target in self.test_loader:
            if self.gpu_mode:
                data, target = data.cuda(self.device), target.cuda(self.device)
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.Classifier(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            for i in range(target.data.shape[0]):
                if pred[i].cpu()[0] == target.data[i]:
                    classe_prediction[pred[i].cpu()[0]] += 1
                else:
                    classe_wrong[pred[i].cpu()[0]] += 1
                classe_total[target.data[i]] += 1

        test_loss /= len(self.test_loader.dataset)
        print(str(self.sigma) + '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
        if not self.conditional:
            #for i in range(10):
            #    print('Classe {} Accuracy: {}/{} ({:.3f}%, Wrong : {})'.format(
            #        i, classe_prediction[i], classe_total[i],
            #        100. * classe_prediction[i] / classe_total[i], classe_wrong[i]))
            #print('\n')
            return test_loss, np.float(correct) / len(self.test_loader.dataset), 100. * classe_prediction / classe_total
        else:
            return test_loss, np.float(correct) / len(self.test_loader.dataset), 100. * classe_prediction[0] / classe_total[0]

    def visualize_Samples(self):
        print("some sample from the generator")
        data, target = self.generator.sample(self.batch_size)
        
        
        dir_path = self.sample_dir + '/' + self.dataset + '/' + self.model_name + '/num_examples_' + str(
            self.num_examples)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        
        if self.gpu_mode:
            data = data.cpu().numpy().transpose(0, 2, 3, 1)
        else:
            data = data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(data[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          dir_path + '/' + self.model_name + '_NumExample%03d' % self.num_examples + '.png')


    def compute_KLD(self):
        self.load(reference=True)
        self.reference_classifier = copy.deepcopy(self.Classifier)
        self.load(reference=False)  # reload the best classifier of the generator

        self.reference_classifier.eval()
        self.Classifier.eval()
        kld = 0
        KLDiv = torch.nn.KLDivLoss()
        KLDiv.size_average = False
        for data, target in self.test_loader:
            if self.gpu_mode:
                data, target = data.cuda(self.device), target.cuda(self.device)
            data, target = Variable(data, volatile=True), Variable(target)
            P = self.reference_classifier(data)
            Q = self.Classifier(data)

            kld += KLDiv(Q, torch.exp(P)).data.cpu()[0]
        print("Mean KLD : {} \n".format(kld / (len(self.test_loader.dataset))))

    def save(self, best=False):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name, 'num_examples_' + str(self.num_examples))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if best:
            torch.save(self.Classifier.state_dict(), os.path.join(save_dir, self.model_name + '_Classifier_Best.pkl'))
        else:
            torch.save(self.Classifier.state_dict(), os.path.join(save_dir, self.model_name + '_Classifier.pkl'))

            # with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            #    pickle.dump(self.train_hist, f)

    def load(self, reference=False):
        if reference:
            save_dir = os.path.join(self.save_dir, self.dataset, "Classifier", 'num_examples_' + str(self.num_examples))
            self.Classifier.load_state_dict(torch.load(os.path.join(save_dir, 'Classifier_Classifier_Best.pkl')))
        else:
            save_dir = os.path.join(self.save_dir, self.dataset, self.model_name,
                                    'num_examples_' + str(self.num_examples))
            self.Classifier.load_state_dict(
                torch.load(os.path.join(save_dir, self.model_name + '_Classifier_Best.pkl')))
