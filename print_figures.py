import numpy as np
import os
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler
from itertools import cycle

from matplotlib.ticker import NullFormatter  # useful for `logit` scale


# plot based on all results !
def compute_sum(saveDir, dataset):
    style_c = cycle(['-', '--', ':', '-.'])
    tau = 0.125
    print('##############    '+dataset+'       ##############')
    liste_model = ["VAE", "WGAN", "CVAE", "CGAN"]  # ,"Classifier","WGAN","VAE","ACGAN"]
    liste = [100, 500, 1000, 5000, 10000, 50000]
    liste_seed = [1, 2, 3, 4, 5, 6, 7, 8]

    baseline = []
    baseline_gaussien = []
    baseline_mask = []
    for j in liste:
        name2 = os.path.join(saveDir, dataset, 'Classifier', 'num_examples_' + str(j),
                             'best_score_classif_ref' + dataset + '.txt')
        baseline.append(np.array(np.loadtxt(name2)).max())
        # Load baseline gauss
        name_sigma = os.path.join(saveDir, dataset, 'Classifier', 'num_examples_' + str(j),
                                  'best_score_classif_sigma_0.15' + dataset + '.txt')
        # dirty fix
        if j == "50000":
            name_mask = os.path.join(saveDir, dataset, 'Classifier', 'num_examples_' + str(j),
                                     'best_score_classif_tresh_0.95' + dataset + '.txt')
        else:
            name_mask = os.path.join(saveDir, dataset, 'Classifier', 'num_examples_' + str(j),
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
        save_dir = os.path.join(saveDir, dataset, model)
        all_values = []
        max_tau = []
        for j in liste:
            # save_dir.append(os.path.join(save_dir, dataset, model_name, 'num_examples_' + str(i)))
            values = []
            for i in range(1, 9):
                name_file = os.path.join(save_dir, 'num_examples_' + str(j),
                                         'best_score_classif_' + dataset + '-tau' + str(i * tau) + '.txt')
                values.append(np.loadtxt(name_file))  # [train_loss, train_acc, test_loss, test_acc]
            values = np.array(values)
            max_tau.append(values.max())


            all_values.append(values)
        max_tau = np.array(max_tau)

        print(model)
        print((max_tau - baseline).sum())
    plt.clf()

def plot_sigma_noise(save_dir, dataset, noise_name):
    style_c = cycle(['-', '--', ':', '-.'])
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

    plt.clf()


def plot_num_training_std(save_dir, dataset, model_name, sigma=False):
    style_c = cycle(['-', '--', ':', '-.'])
    tau = 0.125

    # save_dir=[]
    liste = [100, 500, 1000, 5000, 10000, 50000]
    liste_seed = [1, 2,3,4,5,6,7,8]

    val_all_seed=[]
    baseline_all_seed=[]

    for s in liste_seed:
        save_dir2 = os.path.join(save_dir, dataset, model_name)
        baseline_gaussien = []
        baseline = []
        val = []
        for j in liste:

            # Load baseline
            name2 = os.path.join(save_dir, dataset, 'Classifier', 'num_examples_' + str(j), 'seed_' + str(s),
                                 'best_score_classif_ref' + dataset + '.txt')
            baseline.append(np.array(np.loadtxt(name2)).max())
            # Load baseline gauss
            if sigma:
                name2 = os.path.join(save_dir, dataset, 'Classifier', 'num_examples_' + str(j), 'seed_' + str(s),
                                     'best_score_classif_sigma_0.15' + dataset + '.txt')
                baseline_gaussien.append(np.array(np.loadtxt(name2)).max())

            files = []
            values = []
            for i in range(1, 9):
                name = os.path.join(save_dir2, 'num_examples_' + str(j),'seed_'+str(s),
                                    'best_score_classif_' + dataset + '-tau' + str(i * tau) + '.txt')
                files.append(name)
                values.append(np.loadtxt(name))  # [train_loss, train_acc, test_loss, test_acc]
            all_values = np.array(values)
            val.append(all_values)


        val = np.asarray(val)
        baseline = np.asarray(baseline)

        val_all_seed.append(val)
        baseline_all_seed.append(baseline)

    val_all_seed = np.asarray(val_all_seed)
    baseline_all_seed = np.asarray(baseline_all_seed)




    print(val.shape)
    print(baseline.shape)
    print(val_all_seed.shape)
    print(baseline_all_seed.shape)

    mean_val = val_all_seed.mean(0)
    mean_baseline = baseline_all_seed.mean(0)
    std_val = val_all_seed.std(0)
    std_baseline = baseline_all_seed.std(0)




    #plt.plot(liste, mean_baseline-mean_baseline, linewidth=2, label='Baseline', linestyle=next(style_c))
    #plt.fill_between(liste, std_baseline, - std_baseline, alpha=0.5)
    #plt.errorbar(liste, mean_baseline-mean_baseline, yerr=std_baseline, fmt='o')
    for i in range(8):

        plt.subplot(2,4,i+1)
        #plt.xlabel("Num Example")
        #plt.ylabel("Test accuracy")
        plt.xscale('log')
        plt.legend(loc=1, title='tau')
        plt.title('Tau = ' + str(tau * (i + 1)))
        plt.plot(liste, mean_baseline - mean_baseline, linewidth=2, label='Baseline', linestyle=next(style_c))
        plt.fill_between(liste, std_baseline, - std_baseline, alpha=0.5)
        #plt.errorbar(liste, mean_baseline - mean_baseline, yerr=std_baseline, fmt='o')
        mean_val[:, i] = mean_val[:, i]-mean_baseline
        plt.plot(liste, mean_val[:, i], label=str(tau * (i + 1)), linestyle=next(style_c))
        plt.fill_between(liste, mean_val[:, i]+std_val[:, i],mean_val[:, i] - std_val[:, i], alpha=0.5)
        #plt.errorbar(liste, mean_val[:, i], yerr=std_val[:, i], fmt='o')

        plt.axis('off')
    save_dir = "Figures_Paper"

    #plt.title('Test accuracy for ' + model_name)
    print(os.path.join(save_dir,"num_images", dataset + '_' + model_name + '_num_test_accuracy.png'))
    dir_path = os.path.join(save_dir,"num_images")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(os.path.join(dir_path, dataset + '_' + model_name + '_num_test_accuracy.png'))

    plt.clf()


def plot_num_training(save_dir, dataset, model_name, sigma=False):
    style_c = cycle(['-', '--', ':', '-.'])
    tau = 0.125

    # save_dir=[]
    liste = [100, 500, 1000, 5000, 10000, 50000]
    liste_seed = [1, 2,3,4,5,6,7,8]

    val_all_seed=[]
    baseline_all_seed=[]

    for s in liste_seed:
        save_dir2 = os.path.join(save_dir, dataset, model_name)
        baseline_gaussien = []
        baseline = []
        val = []
        for j in liste:

            # Load baseline
            name2 = os.path.join(save_dir, dataset, 'Classifier', 'num_examples_' + str(j), 'seed_' + str(s),
                                 'best_score_classif_ref' + dataset + '.txt')
            baseline.append(np.array(np.loadtxt(name2)).max())
            # Load baseline gauss
            if sigma:
                name2 = os.path.join(save_dir, dataset, 'Classifier', 'num_examples_' + str(j), 'seed_' + str(s),
                                     'best_score_classif_sigma_0.15' + dataset + '.txt')
                baseline_gaussien.append(np.array(np.loadtxt(name2)).max())

            files = []
            values = []
            for i in range(1, 9):
                name = os.path.join(save_dir2, 'num_examples_' + str(j),'seed_'+str(s),
                                    'best_score_classif_' + dataset + '-tau' + str(i * tau) + '.txt')
                files.append(name)
                values.append(np.loadtxt(name))  # [train_loss, train_acc, test_loss, test_acc]
            all_values = np.array(values)
            val.append(all_values)


        val = np.asarray(val)
        baseline = np.asarray(baseline)

        val_all_seed.append(val)
        baseline_all_seed.append(baseline)

    val_all_seed = np.asarray(val_all_seed)
    baseline_all_seed = np.asarray(baseline_all_seed)




    print(val.shape)
    print(baseline.shape)
    print(val_all_seed.shape)
    print(baseline_all_seed.shape)

    mean_val = val_all_seed.mean(0)
    mean_baseline = baseline_all_seed.mean(0)
    std_val = val_all_seed.std(0)
    std_baseline = baseline_all_seed.std(0)


    plt.plot(liste, mean_baseline-mean_baseline, linewidth=2, label='Baseline', linestyle=next(style_c))
    #plt.fill_between(liste, std_baseline, - std_baseline, alpha=0.5)
    #plt.errorbar(liste, mean_baseline-mean_baseline, yerr=std_baseline, fmt='o')
    for i in range(8):
        mean_val[:, i] = mean_val[:, i]-mean_baseline
        plt.plot(liste, mean_val[:, i], label=str(tau * (i + 1)), linestyle=next(style_c))
        #plt.fill_between(liste, mean_val[:, i]+std_val[:, i],mean_val[:, i] - std_val[:, i], alpha=0.5)
        #plt.errorbar(liste, mean_val[:, i], yerr=std_val[:, i], fmt='o')

    plt.xlabel("Num Example")
    plt.ylabel("Test accuracy")
    plt.xscale('log')
    plt.legend(loc=1, title='tau')
    plt.title('Test accuracy for ' + model_name)
    save_dir = "Figures_Paper"
    print(os.path.join(save_dir,"num_images", dataset + '_' + model_name + '_num_test_accuracy.png'))
    dir_path = os.path.join(save_dir,"num_images")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(os.path.join(dir_path, dataset + '_' + model_name + '_num_test_accuracy.png'))

    plt.clf()


def plot_tau_training(save_dir, dataset, model_name):
    style_c = cycle(['-', '--', ':', '-.'])
    tau = 0.125
    liste_seed = [1, 2,3,4,5,6,7,8]
    liste = [100, 500, 1000, 5000, 10000, 50000]

    val_all_seed = []
    #baseline_all_seed = []

    save_dir2 = os.path.join(save_dir, dataset, model_name)
    for s in liste_seed:
        all_values=[]
        for j in liste:

            files = []
            values = []

            name2 = os.path.join(save_dir, dataset, 'Classifier', 'num_examples_' + str(j), 'seed_' + str(s),
                                 'best_score_classif_ref' + dataset + '.txt')

            values.append(np.loadtxt(name2))

            for i in range(1, 9):
                name = os.path.join(save_dir2, 'num_examples_' + str(j), 'seed_' + str(s),
                                    'best_score_classif_' + dataset + '-tau' + str(i * tau) + '.txt')
                files.append(name)
                values.append(np.loadtxt(name))  # [train_loss, train_acc, test_loss, test_acc]

            values = np.asarray(values)

            #val_all_seed.append(baseline) # val_all_seed oui
            all_values.append(values)
        all_values = np.asarray(all_values)
        val_all_seed.append(all_values)
    val_all_seed = np.asarray(val_all_seed)
    #baseline_all_seed = np.asarray(baseline_all_seed)

    mean_val = val_all_seed.mean(0)
    #mean_baseline = baseline_all_seed.mean(0)
    std_val = val_all_seed.std(0)
    #std_baseline = baseline_all_seed.std(0)

    print(mean_val.shape)

    for j in range(len(liste)):
        print(mean_val.shape)
        x = np.arange(0, 1.125, 0.125)
        plt.plot(x, mean_val[j], label=liste[j], linestyle=next(style_c))

    plt.xlabel("Tau")
    plt.ylabel("Test accuracy")
    #plt.xscale('log')
    plt.legend(loc=3, title='n')
    plt.title('Test accuracy for ' + model_name)
    save_dir = "Figures_Paper"
    dir_path = os.path.join(save_dir, "tau_images")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print(os.path.join(dir_path, dataset + '_' + model_name + '_tau_test_accuracy.png'))
    plt.savefig(os.path.join(dir_path, dataset + '_' + model_name + '_tau_test_accuracy.png'))

    plt.clf()


def plot_acc_training(saveDir, dataset):
    style_c = cycle(['-', '--', ':', '-.'])
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
            liste = ["100", "500", "1000", "5000", "10000", "50000"]
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

    plt.clf()


def plot_seeds4tau(save_dir, dataset, model_name, tau2print=0, sigma=False):
    style_c = cycle(['-', '--', ':', '-.'])
    tau = 0.125
    print("You will print for tau = ", tau*tau2print)

    # save_dir=[]
    liste = [100, 500, 1000, 5000, 10000, 50000]
    liste_seed = [1, 2, 3, 4, 5, 6, 7, 8]

    val_all_seed = []
    baseline_all_seed = []

    for s in liste_seed:
        save_dir2 = os.path.join(save_dir, dataset, model_name)
        baseline_gaussien = []
        baseline = []
        val = []
        for j in liste:

            # Load baseline
            name2 = os.path.join(save_dir, dataset, 'Classifier', 'num_examples_' + str(j), 'seed_' + str(s),
                                 'best_score_classif_ref' + dataset + '.txt')
            baseline.append(np.array(np.loadtxt(name2)).max())
            # Load baseline gauss
            if sigma:
                name2 = os.path.join(save_dir, dataset, 'Classifier', 'num_examples_' + str(j), 'seed_' + str(s),
                                     'best_score_classif_sigma_0.15' + dataset + '.txt')
                baseline_gaussien.append(np.array(np.loadtxt(name2)).max())

            files = []
            values = []
            for i in range(1, 9):
                name = os.path.join(save_dir2, 'num_examples_' + str(j), 'seed_' + str(s),
                                    'best_score_classif_' + dataset + '-tau' + str(i * tau) + '.txt')
                files.append(name)
                values.append(np.loadtxt(name))  # [train_loss, train_acc, test_loss, test_acc]
            all_values = np.array(values)
            val.append(all_values)

        val = np.asarray(val)
        baseline = np.asarray(baseline)

        val_all_seed.append(val)
        baseline_all_seed.append(baseline)

    val_all_seed = np.asarray(val_all_seed)
    baseline_all_seed = np.asarray(baseline_all_seed)

    print(val.shape)
    print(baseline.shape)
    print(val_all_seed.shape)
    print(baseline_all_seed.shape)

    mean_val = val_all_seed.mean(0)
    mean_baseline = baseline_all_seed.mean(0)
    std_val = val_all_seed.std(0)
    std_baseline = baseline_all_seed.std(0)

    # normalisation
    save_dir = "Figures_Paper"
    dir_path = os.path.join(save_dir, "num_images")
    val_tau_0125 = val_all_seed[:, :, tau2print]
    plt.plot(liste, mean_baseline - mean_baseline, linewidth=2, label='Baseline', linestyle=next(style_c))
    for i in range(1, 8):
        plt.plot(liste, val_tau_0125[i, :] - mean_baseline, label='seed-' + str(i), linestyle=next(style_c))
    plt.xlabel("Num Example")
    plt.ylabel("Test accuracy")
    plt.xscale('log')
    plt.legend(loc=1, title='seed')
    plt.title('Test accuracy for ' + model_name)
    plt.savefig(os.path.join(dir_path, dataset + '_' + model_name + '_num_test_accuracy_tau'+str(tau*tau2print)+'.png'))
    plt.clf()

def plot_classes_training(save_dir, dataset, model_name, sigma=False):
    style_c = cycle(['-', '--', ':', '-.'])
    tau = 0.125

    # save_dir=[]
    liste = [100, 500, 1000, 5000, 10000, 50000]
    liste_seed = [1, 2, 3, 4, 5, 6, 7, 8]

    val_all_seed = []
    baseline_all_seed = []

    for s in liste_seed:
        save_dir2 = os.path.join(save_dir, dataset, model_name)
        baseline = []
        val = []
        for j in liste:

            # Load baseline
            name2 = os.path.join(save_dir, dataset, 'Classifier', 'num_examples_' + str(j), 'seed_' + str(s),
                                 'data_classif_classesref' + dataset + '.txt')
            baseline.append(np.array(np.loadtxt(name2)))

            values = []
            # we take tau =1.0 to better detect bad generator
            name = os.path.join(save_dir2, 'num_examples_' + str(j),'seed_'+str(s),
                                'data_classif_classes' + dataset + '-tau' + str(8 * tau) + '.txt')
            values = np.loadtxt(name)  # [train_loss, train_acc, test_loss, test_acc]
            values = np.array(values)
            val.append(values)

        val = np.asarray(val)
        print(val.shape)


        baseline = np.asarray(baseline)

        val_all_seed.append(val)
        baseline_all_seed.append(baseline)

    val_all_seed = np.asarray(val_all_seed)
    print(val_all_seed.shape)
    baseline_all_seed = np.asarray(baseline_all_seed)




    print("val.shape")
    print(val.shape)
    print("baseline.shape")
    print(baseline.shape)
    print("val_all_seed.shape")
    print(val_all_seed.shape)
    print("baseline_all_seed.shape")
    print(baseline_all_seed.shape)


    for i in range(len(liste)):
        num = liste[i]
        ax =plt.subplot(2, 3, i+1)
        mean_val = val_all_seed.mean(0)
        mean_baseline = baseline_all_seed.mean(0)
        std_val = val_all_seed.std(0)
        std_baseline = baseline_all_seed.std(0)

        N = 10
        ind = np.arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars

        rects1 = ax.bar(ind, mean_val[i], width, color='r', yerr=std_val[i])
        rects2 = ax.bar(ind + width, mean_baseline[i], width, color='b', yerr=std_baseline[i])

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Accuracy')
        ax.set_title('num = '+str(num))
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))

        ax.legend((rects1[0], rects2[0]), ('Generator', 'Baseline'), loc=3)

    save_dir = "Figures_Paper"
    print(os.path.join(save_dir, "classes_images", dataset + '_' + model_name + '_num_test_accuracy.png'))
    dir_path = os.path.join(save_dir, "classes_images")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.suptitle('Classes Test accuracy for ' + model_name)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, dataset + '_' + model_name + '_num_test_accuracy.png'))

    plt.clf()


name_file='logs11_12'

plot_classes_training(name_file, 'mnist', 'VAE')
plot_classes_training(name_file, 'mnist', 'WGAN')
plot_classes_training(name_file, 'fashion-mnist', 'VAE')
plot_classes_training(name_file, 'fashion-mnist', 'WGAN')

'''

plot_tau_training(name_file, 'mnist', 'VAE')
plot_tau_training(name_file, 'mnist', 'WGAN')
plot_tau_training(name_file, 'fashion-mnist', 'VAE')

plot_tau_training(name_file, 'fashion-mnist', 'WGAN')
'''

'''
plot_num_training_std(name_file, 'mnist', 'VAE')
plot_num_training_std(name_file, 'mnist', 'WGAN')
plot_num_training_std(name_file, 'fashion-mnist', 'VAE')
plot_num_training_std(name_file, 'fashion-mnist', 'WGAN')
'''

'''
plot_num_training(name_file, 'mnist', 'CVAE')
plot_num_training(name_file, 'mnist', 'CGAN')
plot_num_training(name_file, 'fashion-mnist', 'CVAE')
plot_num_training(name_file, 'fashion-mnist', 'CGAN')
plot_num_training(name_file, 'mnist', 'VAE')
plot_num_training(name_file, 'mnist', 'WGAN')
plot_num_training(name_file, 'fashion-mnist', 'VAE')
plot_num_training(name_file, 'fashion-mnist', 'WGAN')

plot_tau_training(name_file, 'mnist', 'CVAE')
plot_tau_training(name_file, 'mnist', 'CGAN')
plot_tau_training(name_file, 'fashion-mnist', 'CVAE')
plot_tau_training(name_file, 'fashion-mnist', 'CGAN')


plot_tau_training(name_file, 'mnist', 'VAE')
plot_tau_training(name_file, 'mnist', 'WGAN')
plot_tau_training(name_file, 'fashion-mnist', 'VAE')
plot_tau_training(name_file, 'fashion-mnist', 'WGAN')


plot_acc_training(name_file, 'mnist')
plot_acc_training(name_file, 'fashion-mnist')


plot_sigma_noise(name_file, 'mnist', 'sigma')
plot_sigma_noise(name_file, 'mnist', 'tresh')


compute_sum(name_file, 'mnist')
compute_sum(name_file, 'fashion-mnist')
'''