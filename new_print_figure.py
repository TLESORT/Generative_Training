import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_tau_training(save_dir, dataset, model_name, num_examples):
    tau = 0.125

    #save_dir=[]
    liste=["50","100","500","1000","5000","10000","50000"]
        
    save_dir=os.path.join(save_dir, dataset, model_name)
    max_tau = []
    baseline_gaussien = []
    baseline = []
    #liste=[50,100,500,1000,5000,10000,600000]
    for j in liste:
        #save_dir.append(os.path.join(save_dir, dataset, model_name, 'num_examples_' + str(i)))
        print(j)

        files = []
        values = []
        for i in range(9):
            name = os.path.join(save_dir, 'num_examples_' + j, 'best_score_classif_' + dataset + '-tau' + str(i * tau) + '.txt')
            files.append(name)
            values.append(np.loadtxt(name))  # [train_loss, train_acc, test_loss, test_acc]

        all_values = np.array(values)
        max_tau.append(all_values.max())
        baseline_gaussien.append(all_values[0])
        # Load baseline
        name = os.path.join(save_dir, 'num_examples_' + j, 'best_score_classif_' + dataset + '-tau' + str(-1.0) + '.txt')
        baseline.append(np.array(np.loadtxt(name)).max())
        print(all_values.shape)
        assert all_values.shape[0] == 9

        #max_value = all_values.max(1)

        #assert max_value.shape[1] == 4

        x = np.arange(0, 1.125, 0.125)

        #plt.plot(x, all_values, label='num_examples_' + j)
        
        #plt.legend()
    
    plt.xlabel("num examples")
    plt.ylabel("test accuracy")
    liste=[50,100,500,1000,5000,10000,500000]
    plt.plot(liste, max_tau, label='cvae')
    print(baseline)
    plt.plot(liste, baseline, label='baseline')
    plt.plot(liste, baseline_gaussien, label='baseline gaussian')
    plt.legend()
    plt.show()
    # plt.savefig(os.path.join(save_dir, dataset+'_'+model_name +'_test_accuracy.png'))
    #plt.clf()

    #plt.plot(x, max_value[:, 2])
    #plt.savefig(os.path.join(save_dir, 'test_loss.png'))
    #plt.clf()

    #plt.plot(x, max_value[:, 1])
    #plt.savefig(os.path.join(save_dir, 'train_accuracy.png'))
    #plt.clf()

    #plt.plot(x, max_value[:, 0])
    #plt.savefig(os.path.join(save_dir, 'train_loss.png'))
    #plt.clf()


plot_tau_training('models','mnist','VAE', 50)
