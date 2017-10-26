import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler
from itertools import cycle

style_c = cycle(['-', '--', ':', '-.'])


def plot_tau_training(save_dir, dataset, model_name):
    tau = 0.125

    #save_dir=[]
    liste=["100","500","1000","5000","10000","50000"]
    liste2=[100,500,1000,5000,10000,500000]

    save_dir2=os.path.join(save_dir, dataset, model_name)
    max_tau = []
    baseline_gaussien = []
    baseline = []
    val = []
    #liste=[50,100,500,1000,5000,10000,600000]
    for j in liste:
        # save_dir.append(os.path.join(save_dir, dataset, model_name, 'num_examples_' + str(i)))
        print(j)

        files = []
        values = []
        for i in range(1,9):
            name = os.path.join(save_dir2, 'num_examples_' + j,
                                'best_score_classif_' + dataset + '-tau' + str(i * tau) + '.txt')
            files.append(name)
            values.append(np.loadtxt(name))  # [train_loss, train_acc, test_loss, test_acc]

        all_values = np.array(values)
        val.append(all_values)
        max_tau.append(all_values.max())
        #baseline_gaussien.append(all_values[0])
        # Load baseline
        name2 = os.path.join(save_dir, dataset, 'Classifier', 'num_examples_' + j, 'best_score_classif_ref' + dataset + '.txt')
        baseline.append(np.array(np.loadtxt(name2)).max())
        # Load baseline gauss
        name2 = os.path.join(save_dir, dataset, 'Classifier', 'num_examples_' + j, 'best_score_classif_sigma_0.15' + dataset + '.txt')
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
    for i in range(8):
	plt.plot(liste2, val[:,i], label=str(tau * (i+1)))
    plt.plot(liste2, baseline, linewidth=2, label='Baseline')
    plt.xlabel("Num Example")
    plt.ylabel("Test accuracy")
    plt.xscale('log')
    plt.legend(loc=4, title='tau')
    plt.title('Test accuracy for ' + model_name)
    plt.savefig(os.path.join(save_dir2, dataset+'_'+model_name +'_tau_test_accuracy.png'))
    #plt.clf()
    # plt.ylabel("test accuracy")

    # plt.savefig(os.path.join(save_dir, dataset + '_' + model_name + '_test_accuracy.png'))
    # plt.clf()

    # plt.plot(x, max_value[:, 2])
    # plt.savefig(os.path.join(save_dir, 'test_loss.png'))
    # plt.clf()

    # plt.plot(x, max_value[:, 1])
    # plt.savefig(os.path.join(save_dir, 'train_accuracy.png'))
    # plt.clf()

    # plt.plot(x, max_value[:, 0])
    # plt.savefig(os.path.join(save_dir, 'train_loss.png'))
    # plt.clf()


def plot_num_training(saveDir, dataset):
    tau = 0.125

    # save_dir=[]
    liste_model = ["VAE", "CGAN"]  # ,"Classifier","WGAN","VAE","ACGAN"]
    liste = ["100", "500", "1000", "5000", "10000", "50000"]

    for model in liste_model:
        save_dir = os.path.join(saveDir, dataset, model)
	print(save_dir)
        all_values = []
	max_tau = []
        for j in liste:
            # save_dir.append(os.path.join(save_dir, dataset, model_name, 'num_examples_' + str(i)))
            print(j)

            files = []
            values = []
            for i in range(1,9):
                name_file = os.path.join(save_dir, 'num_examples_' + j, 'best_score_classif_' + dataset + '-tau' + str(i * tau) + '.txt')
                values.append(np.loadtxt(name_file))  # [train_loss, train_acc, test_loss, test_acc]

            values = np.array(values)
	    max_tau.append(values.max())

            print(values.shape)
            #assert values.shape[0] == 8

            # plt.plot(liste, values, label=model+"Tau="+ str(i * tau))

            # plt.legend()



            all_values.append(values)
        liste2=[100,500,1000,5000,10000,500000]
        plt.plot(liste2, max_tau, label=model)
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
    baseline = []
    baseline_gaussien = []
    for j in liste:
	    name2 = os.path.join(saveDir, dataset, 'Classifier', 'num_examples_' + j, 'best_score_classif_ref' + dataset + '.txt')
	    baseline.append(np.array(np.loadtxt(name2)).max())
	    # Load baseline gauss
	    name2 = os.path.join(saveDir, dataset, 'Classifier', 'num_examples_' + j, 'best_score_classif_sigma_0.15' + dataset + '.txt')
	    baseline_gaussien.append(np.array(np.loadtxt(name2)).max())
    plt.plot(liste2, baseline, label='Baseline')
    plt.plot(liste2, baseline_gaussien, label='Baseline + noise')

    plt.xlabel("Num Example")
    plt.ylabel("Test accuracy")
    plt.xscale('log')
    plt.legend(loc=4, title='Model')
    plt.title('Test accuracy with differents models')
    # print(os.path.join(saveDir, dataset+'test_accuracy.png'))
    plt.savefig(os.path.join(saveDir, dataset + 'test_accuracy.png'))


# plot_tau_training('models','mnist','CVAE', 50)

plot_tau_training('models','mnist','VAE')
plt.clf()
plot_tau_training('models','mnist','CGAN')
plt.clf()
plot_num_training('models', 'mnist')
