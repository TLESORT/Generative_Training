import argparse, os
from GAN import GAN
from Classifier import Trainer
from WGAN import WGAN
from VAE import VAE
from BEGAN import BEGAN

import torch
#from ssim import MSSIM

"""parsing and configuration"""


def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--classify', type=bool, default=False)
    parser.add_argument('--knn', type=bool, default=False)
    parser.add_argument('--IS', type=bool, default=False)
    parser.add_argument('--train_G', type=bool, default=False)
    parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['GAN', 'Classifier', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN',
                                 'WGAN_GP' 'DRAGAN', 'LSGAN', 'VAE', "CVAE"],
                        help='The type of GAN')  # , required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'celebA', 'cifar10', 'lsun', 'timagenet'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--num_examples', type=int, default=50000, help='The number of examples to use for train')
    parser.add_argument('--dir', type=str, default='./', help='Working directory')
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
    parser.add_argument('--seed', type=int, default=1664)

    return check_args(parser.parse_args())


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

    return args


"""main"""


def main():
    # parse arguments
    args = parse_args()
    seed = args.seed
    torch.manual_seed(seed)



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

    if args.gpu_mode:
        torch.cuda.manual_seed_all(seed)

    print("Use of model {} with dataset {}, tau={}, num_examples={}".format(args.gan_type, args.dataset, args.tau,
                                                                            args.num_examples))

    if args is None:
        exit()
    # declare instance for GAN
    if args.gan_type == 'GAN' or args.gan_type == 'CGAN':
        model = GAN(args)
    elif args.gan_type == 'VAE' or args.gan_type == 'CVAE':
        model = VAE(args)
    elif args.gan_type == 'WGAN':
        model = WGAN(args)
    elif args.gan_type == 'BEGAN':
        model = BEGAN(args)
    elif args.gan_type == 'Classifier':
        print("Just here to train a classic classifier")
    else:
        raise Exception("[!] There is no option for " + args.gan_type)


    # Train the generator to evaluate
    if args.train_G:
        if args.conditional:  # Train one conditional generator for all classes
            model.train_all_classes()
        else:  # Train one generator per class
            model.train()
        print(" [*] Training finished!")
        # visualize generated data in dir_path
        model.visualize_results(args.epoch)
        print(" [*] Testing finished!")

    # Train a deep classifier to evaluate a given generator
    if args.classify:
        print(" [*] Training Classifier!")
        trainer = Trainer(model, args)
        trainer.train_with_generator()

    # Train a reference deep classifier
    if args.gan_type == 'Classifier':
        print(" [*] Training Classic Classifier!")
        trainer = Trainer(None, args)
        trainer.train_classic()

    # Train a knn classifier to evaluate a given generator
    if args.knn:
        print(" [*] Training Classifier!")
        trainer = Trainer(model, args)
        trainer.knn()

    # evaluate with MSSIM
    if args.MSSIM:
        mssim = MSSIM(model, args)
        mssim.test_mssim()

    if args.IS:
        trainer = Trainer(model, args)
        trainer.Inception_score()


if __name__ == '__main__':
    main()
