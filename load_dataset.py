from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_dataset(dataset, batch_size, defaut='flo'):
    if def == "flo":
        path = "/Tmp/bordesfl/"
        fas = True
    else:
        path = "data/"
        fas = False
    if dataset == 'mnist':
        data_loader = DataLoader(datasets.MNIST(path + 'mnist', train=True, download=True,
            transform=transforms.Compose(
                [transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        if fas:
            data_loader = DataLoader(
                datasets.FashionMNIST(path + 'fashion-mnist', train=True, download=True, transform=transforms.Compose(
                    [transforms.ToTensor()])),
                batch_size=batch_size, shuffle=True)
        else:
            data_loader = data.DataLoader(
                    fashion('fashion_data', train=True, download=True, transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    elif dataset == 'cifar10':
        transform = transforms.Compose(
                [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = datasets.CIFAR10(root=path+'cifar10', train=True,
                   download=True, transform=transform)
        data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    elif dataset == 'celebA':
        data_loader = utils.load_celebA(path + 'celebA', transform=transforms.Compose(
            [transforms.CenterCrop(160), transforms.Scale(64), transforms.ToTensor()]), batch_size=batch_size,
            shuffle=True)

    return data_loader


