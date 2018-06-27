import numpy as np
import matplotlib as mpl
import argparse, os

mpl.use('Agg')
import matplotlib.pyplot as plt
from itertools import cycle



def get_results(name_file, model, dataset, liste_seed, liste_num, list_tau, TrainEval=False):
    if TrainEval:
        id_file = 'best_train_score_classif_'
        id_classe_file = 'data_train_classif_classes'
    else:
        id_file = 'best_score_classif_'
        id_classe_file = 'data_classif_classes'

    all_baseline = []
    baseline_classes_all_seed = []
    val_all_seed = []
    val_classes_all_seed = []
    save_dir = os.path.join(name_file, dataset, model)
    for seed in liste_seed:
        baseline = []
        baseline_classes = []
        val = []
        val_classes = []
        for num in liste_num:
            # Load baseline
            if TrainEval:
                name2 = os.path.join(save_dir, 'num_examples_' + str(num), 'seed_' + str(seed),
                                     id_file + dataset + '-tau0.0.txt')
                name_classes2 = os.path.join(save_dir, 'num_examples_' + str(num), 'seed_' + str(seed),
                                             id_classe_file + dataset + '-tau0.0.txt')
            else:
                name2 = os.path.join(name_file, dataset, 'Classifier', 'num_examples_' + str(num), 'seed_' + str(seed),
                                     'best_score_classif_ref' + dataset + '.txt')
                name_classes2 = os.path.join(name_file, dataset, 'Classifier', 'num_examples_' + str(num),
                                             'seed_' + str(seed),
                                             'data_classif_classesref' + dataset + '.txt')
            baseline.append(np.array(np.loadtxt(name2)).max())
            baseline_classes.append(np.array(np.loadtxt(name_classes2)))

            files = []
            files_classes = []
            values = []
            values_classes = []
            for tau in list_tau:
                if tau == 0:
                    continue
                name = os.path.join(save_dir, 'num_examples_' + str(num), 'seed_' + str(seed),
                                    id_file + dataset + '-tau' + str(tau) + '.txt')
                name_classes = os.path.join(save_dir, 'num_examples_' + str(num), 'seed_' + str(seed),
                                            id_classe_file + dataset + '-tau' + str(tau) + '.txt')
                files.append(name)
                files_classes.append(name_classes)

                values.append(np.loadtxt(name))  # [train_loss, train_acc, test_loss, test_acc]
                values_classes.append(np.loadtxt(name_classes))  # [train_loss, train_acc, test_loss, test_acc]

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

    return np.asarray(all_baseline), np.asarray(val_all_seed), np.asarray(baseline_classes_all_seed), np.asarray(
        val_classes_all_seed)


def plot_acc_training(saveDir, dataset, model_list, baseline_all_seed, val_all_seed, list_tau, TrainEval=False):
    style_c = cycle(['-', '--', ':', '-.'])

    mean_baseline = baseline_all_seed.mean(0)
    std_baseline = baseline_all_seed.std(0)
    mean_val = val_all_seed.mean(1)  # mean over seeds
    std_val = val_all_seed.std(1)  # mean over seeds

    x = np.arange(0, 1.125, 0.125)

    for indice_model in range(len(model_list)):
        std_val_model = np.concatenate([[std_baseline], std_val[indice_model]], 1)
        mean_val_model = np.concatenate([[mean_baseline], mean_val[indice_model]], 1)

        print(list_tau)
        print(mean_val_model)

        plt.plot(list_tau, mean_val_model[-1], label=model_list[indice_model], linestyle=next(style_c))
        plt.fill_between(list_tau, mean_val_model[-1] + std_val_model[-1], mean_val_model[-1] - std_val_model[-1],
                         alpha=0.5)
        plt.xlabel("Tau")
        plt.ylabel("Test accuracy")
        # plt.xscale('log')
        plt.legend(loc=3, title='Model')
        plt.title('Test accuracy with differents models')
        # print(os.path.join(saveDir, dataset+'test_accuracy.png'))

    if dataset == "mnist":
        plt.ylim(0.8, 1.0)
    elif dataset == "fashion-mnist":
        plt.ylim(0.6, 1.0)
    else:
        print("Not implemented")
        return

    if TrainEval:
        plt.savefig(os.path.join(saveDir, dataset + '_train_accuracy_var.png'))
    else:
        plt.savefig(os.path.join(saveDir, dataset + '_accuracy_var.png'))

    plt.clf()

    print(val_all_seed.shape)
    # indice of the best perf over random seeds for each tau for each models
    best_seed = val_all_seed[:,:,0,:].argmax(1)

    print(best_seed)

    print(best_seed.shape)

    max_baseline = baseline_all_seed.max(0)

    # print here figure is best result
    for indice_model in range(len(model_list)):

        best_value = val_all_seed[indice_model,:,0,:].max(0) #best result for each tau

        max_val_model = np.concatenate([max_baseline, best_value], 0)
        plt.plot(list_tau, max_val_model, label=model_list[indice_model], linestyle=next(style_c))
        plt.xlabel("Tau")
        plt.ylabel("Test accuracy")
        plt.legend(loc=3, title='Model')
        plt.title('Maximum test accuracy with differents models')


    if TrainEval:
        plt.savefig(os.path.join(saveDir, dataset + '_max_train_accuracy.png'))
    else:
        plt.savefig(os.path.join(saveDir, dataset + '_max_accuracy.png'))

    plt.clf()


def plot_classes_training(save_dir, liste_num, dataset, model_name, baseline_classes_all_seed, val_classes_all_seed,
                          model_list, list_tau, TrainEval=False):
    style_c = cycle(['-', '--', ':', '-.'])

    for indice_model in range(len(model_list)):
        val_model = val_classes_all_seed[indice_model]
        for i in range(len(liste_num)):
            num = liste_num[i]
            ax = plt.subplot(2, 3, indice_model + 1)
            mean_val = val_model.mean(0)
            mean_baseline = baseline_classes_all_seed.mean(0)
            std_val = val_model.std(0)
            std_baseline = baseline_classes_all_seed.std(0)

            N = 10
            ind = np.arange(N)  # the x locations for the groups
            width = 0.5  # the width of the bars

            rects1 = ax.bar(ind, mean_val[i][-1] - mean_baseline[i], width, color='b', yerr=std_val[i][-1])
            # rects2 = ax.bar(ind + width, mean_baseline[i], width, color='r', yerr=std_baseline[i])

            # add some text for labels, title and axes ticks

            if indice_model%3 == 0:
                ax.set_ylabel('Accuracy')
            ax.set_title(model_list[indice_model])
            ax.set_xticks(ind + width / 2)
            ax.set_xticklabels(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
            plt.ylim(-70, 20)

            # ax.legend((rects1[0], rects2[0]), ('Generator', 'Baseline'), loc=3)
            # ax.legend(rects1[0], 'Generator', loc=3)

    # print(os.path.join(save_dir, "classes_images", dataset + '_' + model_name + '_num_test_accuracy.png'))
    # dir_path = os.path.join(save_dir, "classes_images")
    dir_path = save_dir
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # plt.suptitle('Classes Test accuracy for ' + model_name)

    # plt.ylim(-100, 10)
    plt.tight_layout()
    if TrainEval:
        plt.savefig(os.path.join(dir_path, dataset + '_classes_train_accuracy.png'))
    else:
        plt.savefig(os.path.join(dir_path, dataset + '_classes_test_accuracy.png'))

    plt.clf()


def print_knn(save_dir, log_dir, num, dataset, list_seed, list_model, list_tau, k='k'):
    style_c = cycle(['-', '--', ':', '-.'])

    for model in list_model:
        list_all_seed = []
        for seed in list_seed:
            list_all_tau = []
            for tau in list_tau:
                if tau == 0:
                    file = os.path.join(log_dir, dataset, "Classifier", 'num_examples_' + str(num), 'seed_' + str(seed),
                                        'KNN_ref_' + dataset + '.txt')
                else:
                    file = os.path.join(log_dir, dataset, model, 'num_examples_' + str(num), 'seed_' + str(seed),
                                        'best_score_'+str(k)+ 'nn_' + dataset + '-tau' + str(tau) + '.txt')
                values = np.array(np.loadtxt(file))
                list_all_tau.append(values)
            list_all_seed.append(np.array(list_all_tau))


        val_all_seed = np.array(list_all_seed)


        std_val_model = val_all_seed.std(0)
        mean_val_model = val_all_seed.mean(0)

        print(val_all_seed.shape)
        print(mean_val_model.shape)
        assert mean_val_model.shape[0] == len(list_tau)

        if dataset == "mnist":
            plt.ylim(0.8, 1.0)
        elif dataset == "fashion-mnist":
            plt.ylim(0.6, 1.0)
        else:
            print("Not implemented")
            return

        plt.plot(list_tau, mean_val_model, label=model, linestyle=next(style_c))
        plt.fill_between(list_tau, mean_val_model + std_val_model, mean_val_model - std_val_model, alpha=0.5)
        plt.xlabel("Tau")
        plt.ylabel("KNN Test accuracy")

        plt.legend(loc=3, title='Models')
        plt.title('KNN test accuracy for differents models')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, dataset +'_'+str(k)+ 'nn_accuracy.png'))
    plt.clf()


def print_Inception_Score(save_dir, log_dir, num, dataset, list_seed, list_model):
    style_c = cycle(['-', '--', ':', '-.'])

    list_all_model = []
    for model in list_model:
        list_all_seed = []
        for seed in list_seed:
            ref = np.array(np.loadtxt(os.path.join(log_dir, dataset, "Classifier", 'num_examples_' + str(num),
                                                   'seed_' + str(seed), 'Inception_score_ref_' + dataset + '.txt')))
            if model == 'Ref':
                file = os.path.join(log_dir, dataset, "Classifier", 'num_examples_' + str(num), 'seed_' + str(seed),
                                    'Inception_score_ref_' + dataset + '.txt')
            elif model == "train":
                file = os.path.join(log_dir, dataset, "Classifier", 'num_examples_' + str(num), 'seed_' + str(seed),
                                    'Inception_score_train_' + dataset + '.txt')
            else:
                file = os.path.join(log_dir, dataset, model, 'num_examples_' + str(num), 'seed_' + str(seed),
                                    'Inception_score_' + dataset + '.txt')
            values = np.array(np.loadtxt(file))
            list_all_seed.append(values-ref)

        list_all_model.append(list_all_seed)
    val_all_seed = np.array(list_all_model)

    std_val_model = val_all_seed.std(1)
    mean_val_model = val_all_seed.mean(1)

    max_val_model = val_all_seed.max(1)

    print(mean_val_model)

    # plt.plot(list_model, mean_val_model, label=model, linestyle=next(style_c))
    # plt.fill_between(list_model, mean_val_model + std_val_model, mean_val_model - std_val_model, alpha=0.5)

    # ind = np.arange(N)  # the x locations for the groups
    width = 0.5  # the width of the bars

    print(mean_val_model)

    plt.bar(range(len(list_model)), mean_val_model, width, color='b', yerr=std_val_model)
    plt.xticks(range(len(list_model)), list_model)
    plt.xlabel("Models")
    plt.ylabel("Inception Score")

    plt.legend(loc=3, title='Model')
    plt.title('Inception Score for differents models')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, dataset + '_Inception_Score.png'))
    plt.clf()

    return mean_val_model, std_val_model, max_val_model


def print_Frechet_Inception_Distance(save_dir, log_dir, num, dataset, list_seed, list_model):

    list_all_model = []
    for model in list_model:
        list_all_seed = []
        for seed in list_seed:

            if model == "train":
                file = os.path.join(log_dir, dataset, "Classifier", 'num_examples_' + str(num),
                                    'seed_' + str(seed), 'Frechet_Inception_Distance_train_' + dataset + '.txt')
            else:
                file = os.path.join(log_dir, dataset, model, 'num_examples_' + str(num), 'seed_' + str(seed),
                                    'Frechet_Inception_Distance_' + dataset + '.txt')
            values = np.array(np.loadtxt(file))
            list_all_seed.append(values)

        list_all_model.append(list_all_seed)
    val_all_seed = np.array(list_all_model)

    std_val_model = val_all_seed.std(1)
    mean_val_model = val_all_seed.mean(1)
    max_val_model = val_all_seed.min(1) # the best score is the min for FID

    width = 0.5  # the width of the bars

    plt.bar(range(len(list_model)), mean_val_model, width, color='b', yerr=std_val_model)
    plt.xticks(range(len(list_model)), list_model)
    plt.xlabel("Models")
    plt.ylabel("Frechet Inception Distance")

    plt.legend(loc=3, title='Model')
    plt.title('Frechet Inception Distance for differents models')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, dataset + '_Frechet_Inception_Distance.png'))
    plt.clf()

    return mean_val_model, std_val_model, max_val_model

def fitting_capacity(list_model, dataset, baseline, list_val_tot):
    #acc(tau=0)-acc(tau=1)

    print(dataset)
    #print(baseline.shape) #[seed,num]
    #print(list_val_tot.shape) #[model, seed,num,tau]

    assert baseline[:, 0].shape[0] == 8
    mean_baseline = baseline[:,0].mean()
    print("mean baseline : " + str(mean_baseline))

    print(list_val_tot.shape)
    for i in range(len(list_model)):
        print(list_model[i])
        #print(baseline_all_seed[i].shape)
        print("Fitting Capacity : " + str(list_val_tot[i,:,0,-1].mean()))#-mean_baseline))

def comparatif(list_model, dataset, list_val_tot, FID_max, IS_max):

    width = 0.2  # the width of the bars

    # OURS

    score_max = list_val_tot[:, :, 0, -1].max(-1) # just to test

    # normalization for mena=0 and std=1
    score_max=(score_max-score_max.mean())/score_max.std()

    rects1=plt.bar(np.array(range(len(list_model))), score_max, width, color='g')


    # FID

    # normalization for mena=0 and std=1
    FID_max=(FID_max-FID_max.mean())/FID_max.std()

    #in FID smaller is better then we multiply it by -1 to have a scale wit "bigger is better" as the other score
    FID_max=-1*FID_max

    rects2=plt.bar(np.array(range(len(list_model)))+width, FID_max, width, color='b')


    # IS
    # normalization for mean=0 and std=1
    IS_max=(IS_max-IS_max.mean())/IS_max.std()

    rects3=plt.bar(np.array(range(len(list_model)))-width, IS_max, width, color='r')
    plt.xticks(range(len(list_model)), list_model)


    plt.xlabel("Models")
    plt.ylabel("Normalized results")
    plt.legend((rects1[0], rects2[0],rects3[0]), ('Fitting Capacity', 'FID', "IS"), loc=3)
    plt.title('Comparison of Scores for differents models')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, dataset + '_Comparison_Scores.png'))
    plt.clf()

def comparatif_mean(list_model, dataset, list_val_tot, FID_mean, FID_std, IS_mean, IS_std):

    width = 0.2  # the width of the bars

    # OURS

    score_mean = list_val_tot[:, :, 0, -1].mean(-1)
    score_std = list_val_tot[:, :, 0, -1].std(-1)

    # normalization for mena=0 and std=1
    score_mean = (score_mean - score_mean.mean()) / score_mean.std()
    score_std = (score_std) / score_std.std()

    rects1 = plt.bar(np.array(range(len(list_model))), score_mean, width, color='g', yerr=score_std)
    plt.xticks(range(len(list_model)), list_model)
    plt.xlabel("Models")
    plt.ylabel("Frechet Inception Distance")

    # FID

    # normalization for mena=0 and std=1
    FID_mean = -1 * FID_mean
    FID_mean = (FID_mean - FID_mean.mean()) / FID_mean.std()
    FID_std = (FID_std) / FID_std.std()


    # in FID smaller is better then we multiply it by -1 to have a scale wit "bigger is better" as the other score


    rects2 = plt.bar(np.array(range(len(list_model))) + width, FID_mean, width, color='b', yerr=FID_std)
    plt.xticks(range(len(list_model)), list_model)
    plt.xlabel("Models")
    plt.ylabel("Frechet Inception Distance")

    # IS
    # normalization for mena=0 and std=1

    IS_mean = (IS_mean - IS_mean.mean()) / IS_mean.std()
    IS_std = (IS_std) / IS_std.std()

    rects3 = plt.bar(np.array(range(len(list_model))) - width, IS_mean, width, color='r', yerr=IS_std)
    plt.xticks(range(len(list_model)), list_model)

    plt.xlabel("Models")
    plt.ylabel("Inception Score")
    plt.legend((rects1[0], rects2[0], rects3[0]), ('Ours', 'FID', "IS"), loc=3)
    plt.title('Comparison of Scores for differents models')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, dataset + '_mean_Comparison_Scores.png'))

    plt.clf()


def plot_diagram(saveDir, list_model, dataset, baseline, list_val_tot):


    #for i in range(len(list_model)):
    # Create a figure instance
    if dataset == "mnist":
        plt.ylim(0.8, 1.0)
    elif dataset == "fashion-mnist":
        plt.ylim(0.5, 1.0)
    else:
        print("Not implemented")
        return
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot

    print(list_val_tot[:, :, 0, -1].shape)

    print(baseline[:].shape)
    data=np.array([baseline[:,0], list_val_tot[:, :, 0, -1]])

    data=np.concatenate((baseline[:].transpose(), list_val_tot[:, :, 0, -1]))

    print(data.shape)
    bp = ax.boxplot(data.T)


    print(['Baseline']+list_model)
    ax.set_xticklabels(['Baseline']+list_model)



    # Save the figure
    fig.savefig(os.path.join(saveDir, dataset + '_diagram.png'), bbox_inches='tight')

    plt.clf()

def best_perf(saveDir, list_model, dataset, baseline, list_val_tot):


    print(dataset)

    assert baseline[:, 0].shape[0] == 8
    max_baseline = baseline[:, 0].max()
    mean_baseline = baseline[:, 0].mean()
    print("Max baseline : " + str(max_baseline))
    print("Mean baseline : " + str(mean_baseline))

    print(list_val_tot.shape)
    for i in range(len(list_model)):
        print(list_model[i])
        max = list_val_tot[i, :, 0, -1].max()
        mean = list_val_tot[i, :, 0, -1].mean()
        print(" Max : " + str(max))
        print(" Mean : " + str(mean))




def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--knn', type=bool, default=False)
    parser.add_argument('--IS', type=bool, default=False)
    parser.add_argument('--FID', type=bool, default=False)
    parser.add_argument('--BestPerf', type=bool, default=False)
    parser.add_argument('--others', type=bool, default=False)
    parser.add_argument('--TrainEval', type=bool, default=False)
    parser.add_argument('--Accuracy', type=bool, default=False)
    parser.add_argument('--FittingCapacity', type=bool, default=False)
    parser.add_argument('--Diagram', type=bool, default=False)
    parser.add_argument('--log_dir', type=str, default='logs', help='Logs directory')
    parser.add_argument('--save_dir', type=str, default='Figures_Paper', help='Figures directory')
    parser.add_argument('--comparatif', type=bool, default=False)

    return parser.parse_args()


log_dir = 'logs'
log_dir = '/slowdata/tim_bak/Generative_Model/logs'
save_dir = "Figures_Paper"
#save_dir = "/slowdata/tim_bak/Generative_Model/Figures_Paper"
args = parse_args()
# save_dir = "Figures_Paper/Sans_CGAN"

tau = 0.125

liste_num = [100, 500, 1000, 5000, 10000, 50000]
liste_num = [50000]
liste_seed = [1, 2, 3, 4, 5, 6, 7, 8]
list_val_tot = []
list_val_classes_tot = []

baseline_tot = []
baseline_classes_tot = []

baseline_all_seed = None
baseline_classes = None

list_tau = np.array(range(9)) * tau
#list_tau = [0.0, 1.0]
print(list_tau)

list_model = ['VAE', 'WGAN', 'CGAN', 'CVAE', 'GAN', "BEGAN"]

list_dataset = ['mnist','fashion-mnist']

for model in list_model:
    list_val = []
    list_val_classes = []
    list_baseline = []
    list_baseline_classes = []
    for dataset in list_dataset:
        baseline_all_seed, val_all_seed, baseline_classes, val_classes_all_seed = get_results(log_dir,
                                                                                              model,
                                                                                              dataset,
                                                                                              liste_seed,
                                                                                              liste_num,
                                                                                              list_tau,
                                                                                              args.TrainEval)
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

    list_val = np.array(list_val)
    list_val_tot.append(list_val)
    list_val_classes = np.array(list_val_classes)
    list_val_classes_tot.append(list_val_classes)

baseline_tot = np.array(list_baseline)
baseline_classes_tot = np.array(list_baseline_classes)
list_val_tot = np.array(list_val_tot)
list_val_classes_tot = np.array(list_val_classes_tot)

print(baseline_tot.shape)

print('---------------------------------------------------------------------')
print("baseline_tot \n [dataset, seed, num]")
print(baseline_tot.shape)
print("list_val_tot \n [model, dataset, seed, num, tau]")
print(list_val_tot.shape)
print("baseline_classes_tot \n [model, dataset, seed, num, classes]")
print(baseline_classes_tot.shape)
print("list_val_classes_tot \n [model, dataset, seed, num, tau, classes]")
print(list_val_classes_tot.shape)
print('---------------------------------------------------------------------')



print("best baseline mnist")
print(np.argmax(baseline_tot[0,:,0]))
print("best baseline fashion-mnist")
print(np.argmax(baseline_tot[1,:,0]))

if args.knn:
    for dataset in list_dataset:
        for num in liste_num:
            print_knn(save_dir, log_dir, num, dataset, liste_seed, list_model, list_tau)

if args.IS:
    list_model = ['train', 'VAE', 'WGAN', 'CGAN', 'CVAE', 'GAN', "BEGAN"]
    for dataset in list_dataset:
        for num in liste_num:
            print_Inception_Score(save_dir, log_dir, num, dataset, liste_seed, list_model)

if args.FID:
    list_model = ['train', 'VAE', 'WGAN', 'CGAN', 'CVAE', 'GAN', "BEGAN"]
    for dataset in list_dataset:
        for num in liste_num:
            print_Frechet_Inception_Distance(save_dir, log_dir, num, dataset, liste_seed, list_model)

if args.Accuracy:
    for ind_dataset in range(len(list_dataset)):
        dataset = list_dataset[ind_dataset]
        baseline = baseline_tot[ind_dataset]
        baseline_classes = baseline_classes_tot[ind_dataset]
        plot_acc_training(save_dir, dataset, list_model, baseline, list_val_tot[:, ind_dataset, :, :, :], list_tau,
                          args.TrainEval)

        plot_classes_training(save_dir, liste_num, dataset, model, baseline_classes,
                              list_val_classes_tot[:, ind_dataset, :, :, :], list_model, list_tau, args.TrainEval)

    print(list_val_tot.shape)

if args.FittingCapacity:
    for ind_dataset in range(len(list_dataset)):
        dataset = list_dataset[ind_dataset]
        baseline = baseline_tot[ind_dataset]
        fitting_capacity(list_model, dataset, baseline,  list_val_tot[:, ind_dataset, :, :, :])

if args.Diagram:
    for ind_dataset in range(len(list_dataset)):
        dataset = list_dataset[ind_dataset]
        baseline = baseline_tot[ind_dataset]
        plot_diagram(save_dir, list_model, dataset, baseline,  list_val_tot[:, ind_dataset, :, :, :])


if args.BestPerf:
    for ind_dataset in range(len(list_dataset)):
        dataset = list_dataset[ind_dataset]
        baseline = baseline_tot[ind_dataset]
        best_perf(save_dir, list_model, dataset, baseline, list_val_tot[:, ind_dataset, :, :, :])


if args.comparatif:
    #list_model = ['train', 'VAE', 'WGAN', 'CGAN', 'CVAE', 'GAN', "BEGAN"]
    for ind_dataset in range(len(list_dataset)):
        dataset = list_dataset[ind_dataset]
        for num in liste_num:
            FID_mean, FID_std, FID_max = print_Frechet_Inception_Distance(save_dir, log_dir, num, dataset, liste_seed, list_model)
            IS_mean, IS_std, IS_max = print_Inception_Score(save_dir, log_dir, num, dataset, liste_seed, list_model)


            comparatif(list_model, dataset, list_val_tot[:, ind_dataset, :, :, :], FID_max, IS_max)
            comparatif_mean(list_model, dataset, list_val_tot[:, ind_dataset, :, :, :], FID_mean, FID_std, IS_mean, IS_std)