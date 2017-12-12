from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.utils import data
from fashion import fashion

def load_dataset(dataset, batch_size=64, num_examples=50000, defaut='tim'):
    batch_size_valid = 512
    if defaut == "flo":
        path = "/Tmp/bordesfl/"
        fas = True
    else:
        path = "./data"
        fas = False
    if dataset == 'mnist':
        dataset = datasets.MNIST(path + 'mnist', train=True, download=True, transform=transforms.ToTensor())
        data_loader_train = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(range(num_examples)))
        data_loader_valid = DataLoader(dataset, batch_size=batch_size_valid, sampler=SubsetRandomSampler(range(50000, 60000)))
    elif dataset == 'fashion-mnist':
        if fas:
            dataset = datasets.FashionMNIST(path + 'fashion-mnist', train=True, download=True, transform=transforms.ToTensor())
            data_loader_train = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(range(num_examples)))
            data_loader_valid = DataLoader(dataset, batch_size=batch_size_valid, sampler=SubsetRandomSampler(range(50000, 60000)))
        else:
            dataset = fashion('fashion_data', train=True, download=True, transform=transforms.ToTensor())
            data_loader_train = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(range(num_examples)))
            data_loader_valid = DataLoader(dataset, batch_size=batch_size_valid, sampler=SubsetRandomSampler(range(50000, 60000)))
    elif dataset == 'cifar10':
        if num_examples > 45000: num_examples=45000 # does not work if num_example > 50000
        transform = transforms.Compose(
                [transforms.ToTensor()])
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10(root=path+'cifar10', train=True, download=True, transform=transform)
        data_loader_train = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(range(num_examples)))
        data_loader_valid = DataLoader(dataset, batch_size=batch_size_valid, sampler=SubsetRandomSampler(range(45000, 50000)))
    elif dataset == 'lsun':
        transform = transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset_train = datasets.LSUN(db_path='./data/LSUN/', classes=['bedroom_train', 'bridge_train', 'church_outdoor_train', 'classroom_train',
                      'conference_room_train', 'dining_room_train', 'kitchen_train',
                      'living_room_train', 'restaurant_train', 'tower_train'],transform=transform)

        dataset_val = datasets.LSUN(db_path='./data/LSUN/', classes=['bedroom_val', 'bridge_val', 'church_outdoor_val', 'classroom_val',
                      'conference_room_val', 'dining_room_val', 'kitchen_val',
                      'living_room_val', 'restaurant_val', 'tower_val'],transform=transform)
        print("size train : ", len(dataset_train))
        print("size val : ", len(dataset_val))

        data_loader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=SubsetRandomSampler(range(num_examples)))
        data_loader_valid = DataLoader(dataset_val, batch_size=batch_size_valid, sampler=SubsetRandomSampler(range(45000, 50000)))


    return data_loader_train, data_loader_valid

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

