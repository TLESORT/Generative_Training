from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.utils import data
from fashion import fashion

def load_dataset(dataset, batch_size, num_examples=60000, defaut='tim'):
    if defaut == "flo":
        path = "/Tmp/bordesfl/"
        fas = True
    else:
        path = "./data"
        fas = False
    if dataset == 'mnist':
        data_loader = DataLoader(datasets.MNIST(path + 'mnist', train=True, download=True,
            transform=transforms.Compose(
                [transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True, sampler=SubsetRandomSampler(range(num_examples)))
    elif dataset == 'fashion-mnist':
        if fas:
            data_loader = DataLoader(
                datasets.FashionMNIST(path + 'fashion-mnist', train=True, download=True, transform=transforms.Compose(
                    [transforms.ToTensor()])),
                batch_size=batch_size, shuffle=True, sampler=SubsetRandomSampler(range(num_examples)))
        else:
            data_loader = data.DataLoader(
                    fashion('fashion_data', train=True, download=True, transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True,  sampler=SubsetRandomSampler(range(num_examples)), num_workers=1, pin_memory=True)
    elif dataset == 'cifar10':
        if num_examples > 50000: num_examples=50000 # does not work if num_example > 50000
        transform = transforms.Compose(
                [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = datasets.CIFAR10(root=path+'cifar10', train=True,
                   download=True, transform=transform)
        data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, sampler=SubsetRandomSampler(range(num_examples)), num_workers=8)
    elif dataset == 'celebA':
        data_loader = utils.load_celebA(path + 'celebA', transform=transforms.Compose(
            [transforms.CenterCrop(160), transforms.Scale(64), transforms.ToTensor()]), batch_size=batch_size, shuffle=True,
            sampler=SubsetRandomSampler(range(num_examples)))

    return data_loader

def load_dataset_test(dataset, batch_size, defaut='tim'):
    if defaut == "flo":
        path = "/Tmp/bordesfl/"
        fas = True
    else:
        path = "./data"
        fas = False
    if dataset == 'mnist':
        data_loader = DataLoader(datasets.MNIST(path + 'mnist', train=False, download=True,
            transform=transforms.Compose(
                [transforms.ToTensor()])),
            batch_size=batch_size)
    elif dataset == 'fashion-mnist':
        if fas:
            data_loader = DataLoader(
                datasets.FashionMNIST(path + 'fashion-mnist', train=False, download=True, transform=transforms.Compose(
                    [transforms.ToTensor()])),
                batch_size=batch_size)
        else:
            data_loader = data.DataLoader(
                    fashion('fashion_data', train=False, download=True, transform=transforms.ToTensor()),
                    batch_size=batch_size, num_workers=1, pin_memory=True)
    elif dataset == 'cifar10':
        transform = transforms.Compose(
                [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = datasets.CIFAR10(root=path+'cifar10', train=False,
                   download=True, transform=transform)
        data_loader = DataLoader(trainset, batch_size=batch_size, num_workers=8)
    elif dataset == 'celebA':
        data_loader = utils.load_celebA(path + 'celebA', transform=transforms.Compose(
            [transforms.CenterCrop(160), transforms.Scale(64), transforms.ToTensor()]), batch_size=batch_size)

    return data_loader

