import numpy as np
import os
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler
from itertools import cycle


def compute_sum(saveDir, dataset):
    tau = 0.125
    print('##############    '+dataset+'       ##############')
    # save_dir=[]
    liste_model = ["VAE", "WGAN", "CVAE", "CGAN"]  # ,"Classifier","WGAN","VAE","ACGAN"]
    liste = ["100", "500", "1000", "5000", "10000", "60000"]
    liste_classif = ["100", "500", "1000", "5000", "10000", "50000"]
    liste2 = [100, 500, 1000, 5000, 10000, 500000]

    baseline = []
    baseline_gaussien = []
    baseline_mask = []
    for j in liste_classif:
        name2 = os.path.join(saveDir, dataset, 'Classifier', 'num_examples_' + j,
                             'best_score_classif_ref' + dataset + '.txt')
        baseline.append(np.array(np.loadtxt(name2)).max())
        # Load baseline gauss
        name_sigma = os.path.join(saveDir, dataset, 'Classifier', 'num_examples_' + j,
                                  'best_score_classif_sigma_0.15' + dataset + '.txt')
        # dirty fix
        if j == "50000":
            name_mask = os.path.join(saveDir, dataset, 'Classifier', 'num_examples_' + j,
                                     'best_score_classif_tresh_0.95' + dataset + '.txt')
        else:
            name_mask = os.path.join(saveDir, dataset, 'Classifier', 'num_examples_' + j,
                                     'best_score_classif_tresh_1.0' + dataset + '.txt')
        baseline_gaussien.append(np.array(np.loadtxt(name_sigma)).max())
        baseline_mask.append(np.array(np.loadtxt(name_mask)).max())
    baseline_gaussien = np.array(baseline_gaussien)
    baseline_mask = np.array(baseline_mask)
    baseline = np.array(baseline)
    print('Gaussian')
    print((baseline_gaussien - baseline).sum())
    print('Random Noise')
    print((baseline_mask - baseline).sum())

    for model in liste_model:
        if model == "CVAE" or model == "CGAN":
            liste = liste_classif
        else:
            liste = ["100", "500", "1000", "5000", "10000", "60000"]
        save_dir = os.path.join(saveDir, dataset, model)
        all_values = []
        max_tau = []
        for j in liste:
            # save_dir.append(os.path.join(save_dir, dataset, model_name, 'num_examples_' + str(i)))
            values = []
            for i in range(1, 9):
                name_file = os.path.join(save_dir, 'num_examples_' + j,
                                         'best_score_classif_' + dataset + '-tau' + str(i * tau) + '.txt')
                values.append(np.loadtxt(name_file))  # [train_loss, train_acc, test_loss, test_acc]
            values = np.array(values)
            max_tau.append(values.max())


            all_values.append(values)
        max_tau = np.array(max_tau)

        print(model)
        print((max_tau - baseline).sum())



def plot_sigma_noise(save_dir, dataset, noise_name):
    noise = 0.125
    model_name="Classifier"
    liste = ["100", "500", "1000", "5000", "10000"]#, "50000"]

    liste2 = [100, 500, 1000, 5000, 10000]#, 500000]

    save_dir2 = os.path.join(save_dir, dataset, "Classifier")
    max_tau = []
    baseline_gaussien = []
    baseline = []
    val = []
    # liste=[50,100,500,1000,5000,10000,600000]
    for j in liste:
        # save_dir.append(os.path.join(save_dir, dataset, model_name, 'num_examples_' + str(i)))
        print(j)

        files = []
        values = []
        for i in range(1, 9):

            if i==0:
                name = os.path.join(save_dir2, 'num_examples_' + j,
                                    'best_score_classif_' + 'ref' + dataset + '.txt')
            else:
                name = os.path.join(save_dir2, 'num_examples_' + j,
                                'best_score_classif_' +noise_name+ '_' + str(i * noise) + dataset  + '.txt')
            files.append(name)
            values.append(np.loadtxt(name))  # [train_loss, train_acc, test_loss, test_acc]

        all_values = np.array(values)
        val.append(all_values)
        print(all_values.shape)
        #:assert all_values.shape[0] == 9

        # max_value = all_values.max(1)

        # assert max_value.shape[1] == 4

    #    x = np.arange(0, 1.125, 0.125)
    val = np.asarray(val)
    print(baseline)
    print(val.shape)
    save_dir="Figures_Paper"
    for i in range(8):
        plt.plot(liste2, val[:, i], label=str(noise * (i + 1)), linestyle=next(style_c))
        plt.xlabel("Num Example")
        plt.ylabel("Test accuracy")
        plt.xscale('log')
        plt.legend(loc=1, title='tau')
        plt.title('Test accuracy for ' + model_name)
        plt.savefig(os.path.join(save_dir, dataset + '_' + model_name + '_'+noise_name+'_test_accuracy.png'))



def plot_num_training(save_dir, dataset, model_name):
    tau = 0.125

    # save_dir=[]
    liste = ["100", "500", "1000", "5000", "10000", "60000"]
    liste_classif = ["100", "500", "1000", "5000", "10000", "50000"]

    if model_name == "CVAE" or model_name == "CGAN":
        liste = liste_classif

    liste2 = [100, 500, 1000, 5000, 10000, 500000]

    save_dir2 = os.path.join(save_dir, dataset, model_name)
    max_tau = []
    baseline_gaussien = []
    baseline = []
    val = []
    # liste=[50,100,500,1000,5000,10000,600000]
    for j in liste:
        # save_dir.append(os.path.join(save_dir, dataset, model_name, 'num_examples_' + str(i)))
        print(j)

        files = []
        values = []
        for i in range(1, 9):
            name = os.path.join(save_dir2, 'num_examples_' + j,
                                'best_score_classif_' + dataset + '-tau' + str(i * tau) + '.txt')
            files.append(name)
            values.append(np.loadtxt(name))  # [train_loss, train_acc, test_loss, test_acc]

        all_values = np.array(values)
        val.append(all_values)
        max_tau.append(all_values.max())
        # baseline_gaussien.append(all_values[0])
        # Load baseline
        if j == '60000': j = '50000'
        name2 = os.path.join(save_dir, dataset, 'Classifier', 'num_examples_' + j,
                             'best_score_classif_ref' + dataset + '.txt')
        baseline.append(np.array(np.loadtxt(name2)).max())
        # Load baseline gauss
        name2 = os.path.join(save_dir, dataset, 'Classifier', 'num_examples_' + j,
                             'best_score_classif_sigma_0.15' + dataset + '.txt')
        baseline_gaussien.append(np.array(np.loadtxt(name2)).max())

        print(all_values.shape)
        #:assert all_values.shape[0] == 9

        # max_value = all_values.max(1)

        # assert max_value.shape[1] == 4

    #    x = np.arange(0, 1.125, 0.125)
    val = np.asarray(val)
    baseline = np.asarray(baseline)
    print(baseline)
    print(val.shape)
    plt.plot(liste2, baseline-baseline, linewidth=2, label='Baseline', linestyle=next(style_c))
    for i in range(8):
        plt.plot(liste2, val[:, i]-baseline, label=str(tau * (i + 1)), linestyle=next(style_c))
    plt.xlabel("Num Example")
    plt.ylabel("Test accuracy")
    plt.xscale('log')
    plt.legend(loc=1, title='tau')
    plt.title('Test accuracy for ' + model_name)
    print(os.path.join(save_dir,"num_images", dataset + '_' + model_name + '_num_test_accuracy.png'))
    save_dir="Figures_Paper"
    plt.savefig(os.path.join(save_dir,"num_images", dataset + '_' + model_name + '_num_test_accuracy.png'))


def plot_tau_training(save_dir, dataset, model_name):
    tau = 0.125

    # save_dir=[]
    liste = ["100", "500", "1000", "5000", "10000", "60000"] #"50",
    liste_classif = ["100", "500", "1000", "5000", "10000", "50000"] #"50",

    if model_name == "CVAE" or model_name == "CGAN":
        liste = liste_classif

    liste2 = [100, 500, 1000, 5000, 10000, 500000]

    save_dir2 = os.path.join(save_dir, dataset, model_name)
    max_tau = []
    baseline_gaussien = []
    baseline = []
    val = []
    # liste=[50,100,500,1000,5000,10000,600000]
    for j in liste:
        # save_dir.append(os.path.join(save_dir, dataset, model_name, 'num_examples_' + str(i)))
        print(j)

        files = []
        values = []

        if j == '60000':
            name2 = os.path.join(save_dir, dataset, 'Classifier', 'num_examples_50000',
                             'best_score_classif_ref' + dataset + '.txt')
        else:
            name2 = os.path.join(save_dir, dataset, 'Classifier', 'num_examples_' + j,
                                 'best_score_classif_ref' + dataset + '.txt')

        #baseline
        values.append(np.loadtxt(name2))
        for i in range(1, 9):
            name = os.path.join(save_dir2, 'num_examples_' + j,
                                'best_score_classif_' + dataset + '-tau' + str(i * tau) + '.txt')
            files.append(name)
            values.append(np.loadtxt(name))  # [train_loss, train_acc, test_loss, test_acc]

        values=np.array(values)
        print(values.shape)
        x = np.arange(0, 1.125, 0.125)
        if j == '60000': j=50000
        plt.plot(x, values, label=j, linestyle=next(style_c))

    plt.xlabel("Tau")
    plt.ylabel("Test accuracy")
    #plt.xscale('log')
    plt.legend(loc=3, title='n')
    plt.title('Test accuracy for ' + model_name)
    save_dir="Figures_Paper"
    print(os.path.join(save_dir,"tau_images", dataset + '_' + model_name + '_tau_test_accuracy.png'))
    plt.savefig(os.path.join(save_dir,"tau_images", dataset + '_' + model_name + '_tau_test_accuracy.png'))


def plot_acc_training(saveDir, dataset):
    tau = 0.125

    # save_dir=[]
    liste_model = ["VAE", "WGAN", "CVAE", "CGAN"]  # ,"Classifier","WGAN","VAE","ACGAN"]
    liste = ["100", "500", "1000", "5000", "10000", "60000"]
    liste_classif = ["100", "500", "1000", "5000", "10000", "50000"]
    liste2 = [100, 500, 1000, 5000, 10000, 500000]

    baseline = []
    baseline_gaussien = []
    baseline_mask = []
    for j in liste_classif:
        name2 = os.path.join(saveDir, dataset, 'Classifier', 'num_examples_' + j,
                             'best_score_classif_ref' + dataset + '.txt')
        baseline.append(np.array(np.loadtxt(name2)).max())
        # Load baseline gauss
        name_sigma = os.path.join(saveDir, dataset, 'Classifier', 'num_examples_' + j,
                             'best_score_classif_sigma_0.15' + dataset + '.txt')
        #dirty fix
        if j=="10000" or j=="50000":
            name_mask = os.path.join(saveDir, dataset, 'Classifier', 'num_examples_' + j,
                                     'best_score_classif_tresh_0.01' + dataset + '.txt')
        else:
            name_mask = os.path.join(saveDir, dataset, 'Classifier', 'num_examples_' + j,
                              'best_score_classif_tresh_0.95' + dataset + '.txt')
        baseline_gaussien.append(np.array(np.loadtxt(name_sigma)).max())
        baseline_mask.append(np.array(np.loadtxt(name_mask)).max())
    baseline_gaussien=np.array(baseline_gaussien)
    baseline_mask=np.array(baseline_mask)
    baseline=np.array(baseline)
    plt.plot(liste2, baseline-baseline, label='Baseline', linestyle=next(style_c))
    plt.plot(liste2, baseline_gaussien-baseline, label='Gaussian', linestyle=next(style_c))
    plt.plot(liste2, baseline_mask-baseline, label='Random Noise', linestyle=next(style_c))

    for model in liste_model:
        if model == "CVAE" or model == "CGAN":
            liste = liste_classif
        else:
            liste = ["100", "500", "1000", "5000", "10000", "60000"]
        save_dir = os.path.join(saveDir, dataset, model)
        print(save_dir)
        all_values = []
        max_tau = []
        for j in liste:
            # save_dir.append(os.path.join(save_dir, dataset, model_name, 'num_examples_' + str(i)))
            print(j)

            files = []
            values = []
            for i in range(1, 9):
                name_file = os.path.join(save_dir, 'num_examples_' + j,
                                         'best_score_classif_' + dataset + '-tau' + str(i * tau) + '.txt')
                values.append(np.loadtxt(name_file))  # [train_loss, train_acc, test_loss, test_acc]

            values = np.array(values)
            max_tau.append(values.max())

            print(values.shape)
            # assert values.shape[0] == 8

            # plt.plot(liste, values, label=model+"Tau="+ str(i * tau))

            # plt.legend()



            all_values.append(values)
        max_tau=np.array(max_tau)
        liste2 = [100, 500, 1000, 5000, 10000, 500000]
        plt.plot(liste2, max_tau-baseline, label=model)
        all_values = np.array(all_values)
        # assert all_values.shape[1] == 8  # num of different tau
        max_value = all_values.max(1)
        print(all_values.shape)

        assert max_value.shape[0] == 6  # num of different n

        # assert max_value.shape[1] == 4

        # x = np.arange(0, 1.125, 0.125)
    """
	    for i in range(all_values.shape[1]):
	        plt.plot(range(7), all_values[:, i], label=model + " Tau=" + str(i * tau), linestyle=next(style_c)) plt.legend()
    """


    plt.xlabel("Num Example")
    plt.ylabel("Test accuracy")
    plt.xscale('log')
    plt.legend(loc=1, title='Model')
    plt.title('Test accuracy with differents models')
    # print(os.path.join(saveDir, dataset+'test_accuracy.png'))
    save_dir="Figures_Paper"
    plt.savefig(os.path.join(save_dir, dataset + 'test_accuracy.png'))

name_file='models_clean'



style_c = cycle(['-', '--', ':', '-.'])
plot_num_training(name_file, 'mnist', 'CVAE')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
plot_num_training(name_file, 'mnist', 'CGAN')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
plot_num_training(name_file, 'fashion-mnist', 'CVAE')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
plot_num_training(name_file, 'fashion-mnist', 'CGAN')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
plot_num_training(name_file, 'mnist', 'VAE')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
plot_num_training(name_file, 'mnist', 'WGAN')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
plot_num_training(name_file, 'fashion-mnist', 'VAE')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
plot_num_training(name_file, 'fashion-mnist', 'WGAN')
plt.clf()

style_c = cycle(['-', '--', ':', '-.'])
plot_tau_training(name_file, 'mnist', 'CVAE')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
plot_tau_training(name_file, 'mnist', 'CGAN')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
plot_tau_training(name_file, 'fashion-mnist', 'CVAE')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
plot_tau_training(name_file, 'fashion-mnist', 'CGAN')
plt.clf()


style_c = cycle(['-', '--', ':', '-.'])
plot_tau_training(name_file, 'mnist', 'VAE')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
plot_tau_training(name_file, 'mnist', 'WGAN')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
plot_tau_training(name_file, 'fashion-mnist', 'VAE')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
plot_tau_training(name_file, 'fashion-mnist', 'WGAN')
plt.clf()


style_c = cycle(['-', '--', ':', '-.'])
plot_acc_training(name_file, 'mnist')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
plot_acc_training(name_file, 'fashion-mnist')
plt.clf()


style_c = cycle(['-', '--', ':', '-.'])
plot_sigma_noise(name_file, 'mnist', 'sigma')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
plot_sigma_noise(name_file, 'mnist', 'tresh')
plt.clf()



style_c = cycle(['-', '--', ':', '-.'])
compute_sum(name_file, 'mnist')
plt.clf()
style_c = cycle(['-', '--', ':', '-.'])
compute_sum(name_file, 'fashion-mnist')
