
#!/usr/bin/env python
"""Module providing functionality to implement Structural Similarity Image 
Quality Assessment. Based on original paper by Z. Whang
"Image Quality Assessment: From Error Visibility to Structural Similarity" IEEE
Transactions on Image Processing Vol. 13. No. 4. April 2004.
"""

# from https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py

import sys
import numpy
from scipy import signal
from scipy import ndimage
import sort_utils
import torch
import torchvision
from torch.utils import data
from torchvision import datasets, transforms
import torch.optim as optim
import utils
from torch.utils.data import DataLoader
import fashion
import numpy as np

import gauss

class MSSIM(object):
    def __init__(self, model, args):
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
        self.lr = args.lrC
        self.momentum = args.momentum
        self.log_interval = 100
        self.size_epoch = 1000
        self.gan_type = args.gan_type
        self.generator = model
        self.conditional = args.conditional
        self.device = args.device
        # Load the generator parameters
        if self.gan_type != "Classifier":
            if self.conditional:
                self.generator.load()
            else:
                self.generators = self.generator.load_generators()

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

            self.train_loader = data.DataLoader(
                fashion('fashion_data', train=True, download=True, transform=transforms.ToTensor()),
                batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
            self.test_loader = data.DataLoader(
                fashion('fashion_data', train=False, download=True, transform=transforms.ToTensor()),
                batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        elif self.dataset == 'celebA':
            self.train_loader = utils.load_celebA('data/celebA', transform=transforms.Compose(
                [transforms.CenterCrop(160), transforms.Scale(64), transforms.ToTensor()]), batch_size=self.batch_size,
                                                 shuffle=True)
        elif self.dataset == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform)
            self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                            shuffle=False, num_workers=2)

            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=transform)
            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                           shuffle=False, num_workers=2)

    def ssim(self,img1, img2, cs_map=False):
        """Return the Structural Similarity Map corresponding to input images img1
        and img2 (images are assumed to be uint8)

        This function attempts to mimic precisely the functionality of ssim.m a
        MATLAB provided by the author's of SSIM
        https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
        """
        img1 = img1.astype(numpy.float64)
        img2 = img2.astype(numpy.float64)
        size = 11
        sigma = 1.5
        window = gauss.fspecial_gauss(size, sigma)
        K1 = 0.01
        K2 = 0.03
        L = 255 #bitdepth of image
        C1 = (K1*L)**2
        C2 = (K2*L)**2
        mu1 = signal.fftconvolve(window, img1, mode='valid')
        mu2 = signal.fftconvolve(window, img2, mode='valid')
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2
        sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
        sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
        sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
        if cs_map:
            return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                        (sigma1_sq + sigma2_sq + C2)),
                    (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
        else:
            return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                        (sigma1_sq + sigma2_sq + C2))

    def msssim(self, img1, img2):
        """This function implements Multi-Scale Structural Similarity (MSSSIM) Image
        Quality Assessment according to Z. Wang's "Multi-scale structural similarity
        for image quality assessment" Invited Paper, IEEE Asilomar Conference on
        Signals, Systems and Computers, Nov. 2003

        Author's MATLAB implementation:-
        http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
        """
        level = 5
        weight = numpy.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        downsample_filter = numpy.ones((2, 2))/4.0
        im1 = img1.astype(numpy.float64)
        im2 = img2.astype(numpy.float64)
        mssim = numpy.array([])
        mcs = numpy.array([])
        for l in range(level):
            ssim_map, cs_map = self.ssim(im1, im2, cs_map=True)
            mssim = numpy.append(mssim, ssim_map.mean())
            mcs = numpy.append(mcs, cs_map.mean())
            filtered_im1 = ndimage.filters.convolve(im1, downsample_filter,
                                                    mode='reflect')
            filtered_im2 = ndimage.filters.convolve(im2, downsample_filter,
                                                    mode='reflect')
            im1 = filtered_im1[::2, ::2]
            im2 = filtered_im2[::2, ::2]
        return (numpy.prod(mcs[0:level-1]**weight[0:level-1])*
                        (mssim[level-1]**weight[level-1]))

    def test_mssim(self):
        list_classes = sort_utils.get_list_batch(self.train_loader)  # list filled all classe sorted by class
        print("this test is not meant for conditional network right now")
        nb_iteration=100
        ms_ssim_gen=np.zeros(10)
        ms_ssim_real=np.zeros(10)
        for iter in range(nb_iteration):
            for classe in range(10):
                batch_gen, target = self.generator.sample(2, classe)
                batch_real = sort_utils.get_batch(list_classes, classe, 2)
                print(batch_gen.data.shape)
                print(batch_real.data.shape)
                img_gen=batch_gen.data.cpu().numpy()
                img_real=batch_real.data.cpu().numpy()
                print(img_gen.shape)
                print(img_real.shape)
                ms_ssim_gen[classe] += self.msssim(img_gen[0], img_gen[1])
                ms_ssim_real[classe] += self.msssim(img_real[0], img_real[1])

        ms_ssim_gen /= nb_iteration
        ms_ssim_real /= nb_iteration

    def main(self):
        """Compute the SSIM index on two input images specified on the cmd line."""
        import pylab
        argv = sys.argv
        if len(argv) != 3:
            print >>sys.stderr, 'usage: python -m sp.ssim image1.tif image2.tif'
            sys.exit(2)

        try:
            from PIL import Image
            img1 = numpy.asarray(Image.open(argv[1]))
            img2 = numpy.asarray(Image.open(argv[2]))
        except Exception, e:
            e = 'Cannot load images' + str(e)
            print >> sys.stderr, e

        ssim_map = self.ssim(img1, img2)
        ms_ssim = self.msssim(img1, img2)

        pylab.figure()
        pylab.subplot(131)
        pylab.title('Image1')
        pylab.imshow(img1, interpolation='nearest', cmap=pylab.gray())
        pylab.subplot(132)
        pylab.title('Image2')
        pylab.imshow(img2, interpolation='nearest', cmap=pylab.gray())
        pylab.subplot(133)
        pylab.title('SSIM Map\n SSIM: %f\n MSSSIM: %f' % (ssim_map.mean(), ms_ssim))
        pylab.imshow(ssim_map, interpolation='nearest', cmap=pylab.gray())
        pylab.show()

        return 0

