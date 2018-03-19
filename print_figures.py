import numpy as np
import os
import matplotlib as mpl
import argparse, os
mpl.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler
from itertools import cycle

from matplotlib.ticker import NullFormatter  # useful for `logit` scale


def get_results(name_file, model, dataset, liste_seed, liste_num):
    all_baseline = []
    baseline_classes_all_seed = []
    val_all_seed = []
    val_classes_all_seed = []
    save_dir = os.path.join(name_file, dataset, model)
    for s in liste_seed:
        baseline = []
        baseline_classes = []
        val = []
        val_classes = []
        for j in liste_num:
            # Load baseline
            name2 = os.path.join(name_file, dataset, 'Classifier', 'num_examples_' + str(j), 'seed_' + str(s),
                                'best_score_classif_ref' + dataset + '.txt')
            name_classes2 = os.path.join(name_file, dataset, 'Classifier', 'num_examples_' + str(j), 'seed_' + str(s),
                                         'data_classif_classesref' + dataset + '.txt')
            baseline.append(np.array(np.loadtxt(name2)).max())
            baseline_classes.append(np.array(np.loadtxt(name_classes2)))

            files = []
            files_classes = []
            values = []
            values_classes = []
            for i in range(1, 9):
                name = os.path.join(save_dir, 'num_examples_' + str(j), 'seed_' + str(s),
                                    'best_score_classif_' + dataset + '-tau' + str(i * tau) + '.txt')
                name_classes = os.path.join(save_dir, 'num_examples_' + str(j), 'seed_' + str(s),
                                        'data_classif_classes' + dataset + '-tau' + str(i * tau) + '.txt')
                print(name)
                files.append(name)
                files_classes.append(name_classes)

                values.append(np.loadtxt(name))  # [train_loss, train_acc, test_loss, test_acc]
                values_classes.append(np.loadtxt(name_classes))  # [train_loss, train_acc, test_loss, test_acc]

            print(values)
            values = np.array(values)
            val.append(values)
            values_classes = np.array(values_classes)
            val_classes.append(values_classes)

        baseline_classes = np.asarray(baseline_classes)
        baseline = np.asarray(baseline)

        val = np.asarray(val)
        val_classes = np.asarray(val_classes)

        val_all_seed.append(val)
        val_classes_all_seed.append(val_classes)
        baseline_classes_all_seed.append(baseline_classes)
        all_baseline.append(baseline)

    return np.asarray(all_baseline), np.asarray(val_all_seed), np.asarray(baseline_classes_all_seed), np.asarray(val_classes_all_seed)


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

        axe=plt.subplot(2,4,i+1)
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

        #lt.axis('off')

    #plt.title('Test accuracy for ' + model_name)
    print(os.path.join(save_dir,"num_images", dataset + '_' + model_name + '_num_test_accuracy.png'))
    dir_path = os.path.join(save_dir,"num_images")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.tight_layout()
    plt.axis('equal')
    plt.savefig(os.path.join(dir_path, dataset + '_' + model_name + '_num_test_accuracy.png'))

    plt.clf()


def plot_num_training(save_dir, dataset, model_name, sigma=False):
    style_c = cycle(['-', '--', ':', '-.'])
    tau = 0.125

    # save_dir=[]
    liste = [100, 500, 1000, 5000, 10000, 50000]
    liste_seed = [1, 2, 3, 4, 5, 6, 7, 8]

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
    print(os.path.join(save_dir,"num_images", dataset + '_' + model_name + '_num_test_accuracy.png'))
    dir_path = os.path.join(save_dir,"num_images")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(os.path.join(dir_path, dataset + '_' + model_name + '_num_test_accuracy.png'))

    plt.clf()


def plot_tau_training(save_dir, dataset, model_name,baseline_all_seed, val_all_seed, liste_num):
    style_c = cycle(['-', '--', ':', '-.'])

    mean_val = val_all_seed.mean(0)
    mean_baseline = baseline_all_seed.mean(0)
    std_val = val_all_seed.std(0)
    std_baseline = baseline_all_seed.std(0)

    print(mean_val.shape)
    print(mean_baseline.shape)
    mean_val = np.concatenate([[mean_baseline], mean_val], 1)
    print(mean_val.shape)

    print(mean_val.shape)

    for j in range(len(liste_num)):
        print(mean_val.shape)
        x = np.arange(0, 1.125, 0.125)
        plt.plot(x, mean_val[j], label=liste_num[j], linestyle=next(style_c))

    plt.xlabel("Tau")
    plt.ylabel("Test accuracy")
    #plt.xscale('log')
    plt.legend(loc=3, title='n')
    plt.title('Test accuracy for ' + model_name)
    dir_path = os.path.join(save_dir, "tau_images")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print(os.path.join(dir_path, dataset + '_' + model_name + '_tau_test_accuracy.png'))
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, dataset + '_' + model_name + '_tau_test_accuracy.png'))

    plt.clf()


def plot_acc_training(saveDir, dataset, model_list,baseline_all_seed , val_all_seed):
    style_c = cycle(['-', '--', ':', '-.'])

    mean_baseline = baseline_all_seed.mean(0)
    std_baseline = baseline_all_seed.std(0)
    mean_val = val_all_seed.mean(1) # mean over seeds
    std_val = val_all_seed.std(1) # mean over seeds

    x = np.arange(0, 1.125, 0.125)

    for indice_model in range(len(model_list)):
        std_val_model = np.concatenate([[std_baseline], std_val[indice_model]], 1)
        mean_val_model = np.concatenate([[mean_baseline], mean_val[indice_model]], 1)
        plt.plot(x, mean_val_model[-1], label=model_list[indice_model], linestyle=next(style_c))
        plt.fill_between(x, mean_val_model[-1] + std_val_model[-1], mean_val_model[-1] - std_val_model[-1], alpha=0.5)
        plt.xlabel("Tau")
        plt.ylabel("Test accuracy")
        #plt.xscale('log')
        plt.legend(loc=3, title='Model')
        plt.title('Test accuracy with differents models')
        # print(os.path.join(saveDir, dataset+'test_accuracy.png'))
    plt.savefig(os.path.join(saveDir, dataset + '_accuracy_var.png'))

    plt.clf()

    for indice_model in range(len(model_list)):
        #std_val_model = np.concatenate([[std_baseline], std_val[indice_model]], 1)
        mean_val_model = np.concatenate([[mean_baseline], mean_val[indice_model]], 1)
        plt.plot(x, mean_val_model[-1], label=model_list[indice_model], linestyle=next(style_c))
        #plt.fill_between(x, mean_val_model[-1] + std_val_model[-1], mean_val_model[-1] - std_val_model[-1], alpha=0.5)
        plt.xlabel("Tau")
        plt.ylabel("Test accuracy")
        #plt.xscale('log')
        plt.legend(loc=3, title='Model')
        plt.title('Test accuracy with differents models')
        # print(os.path.join(saveDir, dataset+'test_accuracy.png'))
    plt.savefig(os.path.join(saveDir, dataset + '_accuracy.png'))

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

def plot_classes_training(save_dir, liste_num, dataset, model_name, baseline_classes_all_seed, val_classes_all_seed,model_list):
    style_c = cycle(['-', '--', ':', '-.'])

    for indice_model in range(len(model_list)):
        val_model=val_classes_all_seed[indice_model]
        for i in range(len(liste_num)):
            num = liste_num[i]
            plt.ylim(-100, 10)
            ax =plt.subplot(2, 3, indice_model+1)
            mean_val = val_model.mean(0)
            mean_baseline = baseline_classes_all_seed.mean(0)
            std_val = val_model.std(0)
            std_baseline = baseline_classes_all_seed.std(0)

            N = 10
            ind = np.arange(N)  # the x locations for the groups
            width = 0.5  # the width of the bars

            print(mean_val[i][7].shape)
            print(mean_baseline.shape)

            rects1 = ax.bar(ind, mean_val[i][7] - mean_baseline[i], width, color='b', yerr=std_val[i][7])
            #rects2 = ax.bar(ind + width, mean_baseline[i], width, color='r', yerr=std_baseline[i])

            # add some text for labels, title and axes ticks
            ax.set_ylabel('Accuracy')
            ax.set_title(model_list[indice_model])
            ax.set_xticks(ind + width / 2)
            ax.set_xticklabels(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))

        #ax.legend((rects1[0], rects2[0]), ('Generator', 'Baseline'), loc=3)
        #ax.legend(rects1[0], 'Generator', loc=3)


    #print(os.path.join(save_dir, "classes_images", dataset + '_' + model_name + '_num_test_accuracy.png'))
    #dir_path = os.path.join(save_dir, "classes_images")
    dir_path = save_dir
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    #plt.suptitle('Classes Test accuracy for ' + model_name)

    #plt.ylim(-100, 10)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, dataset  + '_num_test_accuracy.png'))

    plt.clf()


def print_knn(save_dir,log_dir, num, dataset, list_seed, list_model, list_tau):
    style_c = cycle(['-', '--', ':', '-.'])
    x = np.arange(0, 1.125, 0.125)

    list_all_model = []
    for model in list_model:
        list_all_seed = []
        for seed in list_seed:
            list_all_tau = []
            for tau in list_tau:
                file = os.path.join(log_dir, dataset, model, 'num_examples_' + str(num), 'seed_' + str(seed),
                                     'best_score_knn_' + dataset + '-tau' + str(self.tau) + '.txt')
                values = np.array(np.loadtxt(file))
                list_all_tau.append(values)
            list_all_seed.append(np.array(list_all_tau))

        val_all_seed = np.array(list_all_seed)
        std_val_model = val_all_seed.std(0)
        mean_val_model = val_all_seed.mean(0)

        print(val_all_seed.shape)
        print(mean_val_model.shape)
        assert mean_val_model.shape[0] == len(list_tau)

        plt.plot(list_tau, mean_val_model, label=model, linestyle=next(style_c))
        plt.fill_between(list_tau, mean_val_model + std_val_model, mean_val_model - std_val_model, alpha=0.5)
        plt.xlabel("Tau")
        plt.ylabel("KNN Test accuracy")

        plt.legend(loc=3, title='Models')
        plt.title('KNN test accuracy for differents models')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, dataset + '_knn_accuracy.png'))
    plt.clf()


def print_Inception_Score(save_dir,log_dir, num, dataset, list_seed, list_model):
    style_c = cycle(['-', '--', ':', '-.'])
    x = np.arange(0, 1.125, 0.125)

    list_all_model = []
    for model in list_model:
        list_all_seed = []
        for seed in list_seed:
            file = os.path.join(log_dir, dataset, model, 'num_examples_' + str(num), 'seed_' + str(seed),
                                 'Inception_score_'+dataset+'.txt')
            values = np.array(np.loadtxt(file))
            list_all_seed.append(np.array(values))

        list_all_model.append(list_all_seed)
    val_all_seed = np.array(list_all_model)

    std_val_model = val_all_seed.std(1)
    mean_val_model = val_all_seed.mean(1)

    print(val_all_seed.shape)
    print(mean_val_model.shape)

    #plt.plot(list_model, mean_val_model, label=model, linestyle=next(style_c))
    #plt.fill_between(list_model, mean_val_model + std_val_model, mean_val_model - std_val_model, alpha=0.5)

    #ind = np.arange(N)  # the x locations for the groups
    width = 0.5  # the width of the bars

    plt.bar(range(6), mean_val_model, width, color='b', yerr=std_val_model)

    plt.xlabel("Models")
    plt.ylabel("Inception Score")


    plt.legend(loc=3, title='Model')
    plt.title('Inception Score for differents models')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, dataset + '_Inception_Score.png'))
    plt.clf()


def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--knn', type=bool, default=False)
    parser.add_argument('--IS', type=bool, default=False)
    parser.add_argument('--others', type=bool, default=False)

    return parser.parse_args()




log_dir='logs'
log_dir='/slowdata/tim_bak/Generative_Model/logs'
save_dir = "Figures_Paper"
save_dir = "/slowdata/tim_bak/Generative_Model/Figures_Paper"
args = parse_args()
#save_dir = "Figures_Paper/Sans_CGAN"

tau=0.125

liste_num = [100, 500, 1000, 5000, 10000, 50000]
liste_num = [50000]
liste_seed = [1, 2, 3, 4, 5, 6, 7,8]
liste_seed = [1, 2, 3, 4, 5, 6, 7]

list_tau = np.array(range(10))*tau


list_model = ['VAE', 'WGAN', 'CGAN', 'CVAE', 'GAN', "BEGAN"]

list_dataset = ['fashion-mnist', 'mnist']
#list_dataset = ['fashion-mnist']

if args.knn:
    for dataset in list_dataset:
        for num in liste_num:
            print_knn(save_dir, log_dir, num, dataset, liste_seed, list_model, list_tau)

if args.IS:
    for dataset in list_dataset:
        for num in liste_num:
            print_Inception_Score(save_dir, log_dir, num, dataset, liste_seed, list_model)


if args.others:
    list_val_tot = []
    list_val_classes_tot = []

    baseline_tot = []
    baseline_classes_tot = []

    baseline_all_seed = None
    baseline_classes = None


    for model in list_model:
        list_val = []
        list_val_classes = []
        list_baseline = []
        list_baseline_classes = []
        for dataset in list_dataset:
            baseline_all_seed, val_all_seed, baseline_classes, val_classes_all_seed = get_results(log_dir, model, dataset, liste_seed, liste_num)
            list_val.append(val_all_seed)
            list_val_classes.append(np.array(val_classes_all_seed))

            list_baseline.append(baseline_all_seed)
            list_baseline_classes.append(np.array(baseline_classes))
            print("Model : " + model + " ::: Dataset :" + dataset)
            print("[seed, num]")
            print(baseline_all_seed.shape)
            print("[seed, num, tau]")
            print(val_all_seed.shape)
            print("[seed, num, classes]")
            print(baseline_classes.shape)
            print("[seed, num, tau, classes]")
            print(val_classes_all_seed.shape)
            #plot_classes_training(save_dir, liste_num, dataset, model, baseline_classes, val_classes_all_seed)
            #plot_tau_training(save_dir, dataset, model, baseline_all_seed, val_all_seed, liste_num)

        list_val = np.array(list_val)
        list_val_tot.append(list_val)
        list_val_classes = np.array(list_val_classes)
        list_val_classes_tot.append(list_val_classes)
        baseline_tot = np.array(list_baseline)
        baseline_classes_tot = np.array(list_baseline_classes)

    list_val_tot = np.array(list_val_tot)
    list_val_classes_tot = np.array(list_val_classes_tot)
    '''
    list_val_classes2=np.zeros((len(list_val_classes),8,1,8,10))
    
    
    for i in range(len(list_val_classes)):
        print(np.array(list_val_classes[i]).shape)
        list_val_classes2[i] = np.array(list_val_classes[i])
    '''
    print(baseline_tot.shape)
    for ind_dataset in range(len(list_dataset)):
        dataset=list_dataset[ind_dataset]
        baseline=baseline_tot[ind_dataset]
        baseline_classes=baseline_classes_tot[ind_dataset]
        plot_acc_training(save_dir, dataset, list_model, baseline, list_val_tot[:,ind_dataset,:,:,:])
        plot_classes_training(save_dir, liste_num, dataset, model, baseline_classes,list_val_classes_tot[:,ind_dataset,:,:,:],list_model)

    print(list_val_tot.shape)

    '''
    plot_tau_training(name_file, 'mnist', 'VAE')
    plot_tau_training(name_file, 'mnist', 'WGAN')
    plot_tau_training(name_file, 'fashion-mnist', 'VAE')
    
    plot_tau_training(name_file, 'fashion-mnist', 'WGAN')
    '''


    #plot_num_training_std(name_file, dataset, model)

    '''
    plot_num_training(name_file, dataset, model)
    
    plot_tau_training(name_file, dataset, model)
    
    
    plot_acc_training(name_file, dataset)
    
    
    plot_sigma_noise(name_file, dataset, 'sigma')
    plot_sigma_noise(name_file, 'mnist', 'tresh')
    
    
    compute_sum(name_file, dataset)
    '''