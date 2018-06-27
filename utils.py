import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

"""checking arguments"""


def check_args(args):
    args.save_dir = os.path.join(args.dir, args.save_dir)
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.result_dir = os.path.join(args.dir, args.result_dir)
    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    args.log_dir = os.path.join(args.dir, args.log_dir)
    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    args.sample_dir = os.path.join(args.dir, args.sample_dir)
    # --sample_dir
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    #
    if args.gan_type == "VAE" and args.conditional:
        args.gan_type = "CVAE"
    if args.gan_type == "GAN" and args.conditional:
        args.gan_type = "CGAN"

    args.result_dir = os.path.join(args.result_dir, args.dataset, args.gan_type, 'num_examples_' +
                                   str(args.num_examples), 'seed_' + str(args.seed))
    args.save_dir = os.path.join(args.save_dir, args.dataset, args.gan_type, 'num_examples_' +
                                 str(args.num_examples), 'seed_' + str(args.seed))
    args.log_dir = os.path.join(args.log_dir, args.dataset, args.gan_type, 'num_examples_' +
                                str(args.num_examples), 'seed_' + str(args.seed))
    args.sample_dir = os.path.join(args.sample_dir, args.dataset, args.gan_type, 'num_examples_' +
                                   str(args.num_examples), 'seed_' + str(args.seed))

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    if args.gan_type == "CVAE" or args.gan_type == "CGAN":
        args.conditional = True

    return args

def get_best_baseline(name_dir, dataset, num_examples):
    id_file = 'best_train_score_classif_'
    seed_best_baseline=-1

    name_dir = os.path.join(name_dir, "..", "..", "..", "Classifier",
                                'num_examples_' + str(num_examples))

    liste_seed = [1, 2, 3, 4, 5, 6, 7, 8]

    print("we search best classifier in seeds : " + str(liste_seed))

    all_baseline = []

    for seed in liste_seed:
        name2 = os.path.join(name_dir, 'seed_' + str(seed),
                             'best_score_classif_ref' + dataset + '.txt')

        all_baseline.append(np.array(np.loadtxt(name2)).max())


    baseline = np.asarray(all_baseline)

    print(baseline.shape)
    print(baseline)

    seed_best_baseline = np.argmax(baseline)



    return seed_best_baseline




def load_mnist(dataset):
    data_dir = os.path.join("./data", dataset)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec


def img_stretch(img):
    img = img.astype(float)
    img -= np.min(img)
    img /= np.max(img)+1e-12
    return img


def make_samples_batche(prediction, batch_size, filename_dest):
    plt.figure()
    batch_size_sqrt = int(np.sqrt(batch_size))
    input_channel = prediction[0].shape[0]
    input_dim = prediction[0].shape[1]
    prediction = np.clip(prediction, 0, 1)
    pred = np.rollaxis(prediction.reshape((batch_size_sqrt, batch_size_sqrt, input_channel, input_dim, input_dim)), 2, 5)
    pred = pred.swapaxes(2, 1)
    pred = pred.reshape((batch_size_sqrt*input_dim, batch_size_sqrt*input_dim, input_channel))
    fig, ax = plt.subplots(figsize=(batch_size_sqrt, batch_size_sqrt))
    ax.axis('off')
    ax.imshow(img_stretch(pred), interpolation='nearest')
    ax.grid()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(filename_dest, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    plt.close()


def load_celebA(dir, transform, batch_size, shuffle):
    # transform = transforms.Compose([
    #     transforms.CenterCrop(160),
    #     transform.Scale(64)
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # data_dir = 'data/celebA'  # this path depends on your computer
    dset = datasets.ImageFolder(dir, transform)
    data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

    return data_loader


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path = '', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
