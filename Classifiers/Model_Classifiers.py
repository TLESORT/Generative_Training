
import torch.nn as nn
import torch.nn.functional as F

class Model_Classifier(nn.Module):
    def __init__(self, dataset):
        super(Model_Classifier, self).__init__()

        self.dataset = dataset
        #'mnist', 'fashion-mnist', 'celebA', 'cifar10', 'celebA', 'timagenet'
        if self.dataset == 'mnist':
            self.classifer=Mnist_Classifier()
        elif self.dataset == 'fashion-mnist':
            self.classifer=Fashion_Classifier()
        elif self.dataset == 'celebA':
            self.classifer=LSUN_Classifier()
        elif self.dataset == 'cifar10':
            self.classifer=Cifar10_Classifier()
        elif self.dataset == 'timagenet':
            self.classifer=Timagenet_Classifier()
        else:
            raise ValueError("This classifier is not implemented")

    def get_classifier_model(self):
        return self.classifer




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


class LSUN_Classifier(nn.Module):
    def __init__(self):
        super(LSUN_Classifier, self).__init__()
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


class Timagenet_Classifier(nn.Module):
    def __init__(self):
        super(Timagenet_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 200)

    def forward(self, x, FID=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        if FID:
            return x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)


class Fashion_Classifier(nn.Module):
    def __init__(self):
        super(Fashion_Classifier, self).__init__()
        self.cnn1 = nn.Conv2d(1, 16, kernel_size=5)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(16, 32, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x, FID=False):
        out = self.cnn1(x)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        if FID:
            return out
        out = self.dropout(out)
        out = self.fc1(out)
        return F.log_softmax(out)


class Mnist_Classifier(nn.Module):
    def __init__(self):
        super(Mnist_Classifier, self).__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv2_drop = nn.Dropout2d()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, FID=False):
        x = self.relu(self.maxpool2(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2_drop(self.conv2(x))))
        x = x.view(-1, 320)
        if FID:
            return x
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x)