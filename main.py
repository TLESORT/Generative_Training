import argparse, os
from GAN import GAN  # not necessary anymore
from Classifier import Trainer
from CGAN import CGAN
# from LSGAN import LSGAN
# from DRAGAN import DRAGAN
#from acgan import ACGAN
from WGAN import WGAN
from VAE import VAE
# from WGAN_GP import WGAN_GP
# from infoGAN import infoGAN
# from EBGAN import EBGAN
# from BEGAN import BEGAN

# from ssim import MSSIM

import torch

"""parsing and configuration"""


def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--classify', type=bool, default=False)
    parser.add_argument('--knn', type=bool, default=False)
    parser.add_argument('--train_G', type=bool, default=False)
    parser.add_argument('--gan_type', type=str, default='EBGAN',
                        choices=['GAN', 'Classifier', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN',
                                 'WGAN_GP' 'DRAGAN', 'LSGAN', 'VAE', "CVAE"],
                        help='The type of GAN')  # , required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'celebA', 'cifar10'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--num_examples', type=int, default=60000, help='The number of examples to use for train')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--sample_dir', type=str, default='Samples',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--lrC', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--conditional', type=bool, default=False)
    parser.add_argument('--MSSIM', type=bool, default=False)
    parser.add_argument('--tau', type=float, default=0.0, help='ratio of training data.')
    parser.add_argument('--sigma', type=float, default=0.0, help='Variance of gaussian noise')
    parser.add_argument('--tresh_masking_noise', type=float, default=0.0, help='Variance of gaussian noise')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--size_epoch', type=int, default=1000)

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

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

    return args

    #if args.num_examples > 50000:
    #    print("this amount of data is not authorized : 50000 is the max")
    #    args.num_examples = 50000


"""main"""


def main():
    # parse arguments
    args = parse_args()
    seed = 1664
    torch.manual_seed(seed)

    if args.gan_type == "CVAE":
        args.gan_type = "VAE"
        args.conditional = True

    if args.gpu_mode:
        torch.cuda.manual_seed_all(seed)

    print("Use of model {} with dataset {}, tau={}, num_examples={}".format(args.gan_type, args.dataset, args.tau,
                                                                            args.num_examples))

    if args is None:
        exit()
    # declare instance for GAN
    if args.gan_type == 'GAN':
        model = GAN(args)
    elif args.gan_type == 'VAE':
        model = VAE(args)
    elif args.gan_type == 'CGAN':
        model = CGAN(args)
    elif args.gan_type == 'ACGAN':
        model = ACGAN(args)
    elif args.gan_type == 'infoGAN':
        model = infoGAN(args, SUPERVISED=True)
    elif args.gan_type == 'EBGAN':
        model = EBGAN(args)
    elif args.gan_type == 'WGAN':
        model = WGAN(args)
    elif args.gan_type == 'WGAN_GP':
        model = WGAN_GP(args)
    elif args.gan_type == 'DRAGAN':
        model = DRAGAN(args)
    elif args.gan_type == 'LSGAN':
        model = LSGAN(args)
    elif args.gan_type == 'BEGAN':
        model = BEGAN(args)
    elif args.gan_type == 'Classifier':
        print("Just here to train a classic classifier")
    else:
        raise Exception("[!] There is no option for " + args.gan_type)

    if args.train_G:
        if args.conditional:
            model.train_all_classes()
        else:
            model.train()
        print(" [*] Training finished!")
        # visualize learned generator
        model.visualize_results(args.epoch)
        print(" [*] Testing finished!")

    if args.classify:
        print(" [*] Training Classifier!")
        trainer = Trainer(model, args)
        trainer.train_with_generator()

    if args.knn:
        print(" [*] Training Classifier!")
        trainer = Trainer(model, args)
        trainer.knn()

    if args.gan_type == 'Classifier':
        print(" [*] Training Classic Classifier!")
        trainer = Trainer(None, args)
        trainer.train_classic()

    if args.MSSIM:
        mssim = MSSIM(model, args)
        mssim.test_mssim()


if __name__ == '__main__':
    main()
