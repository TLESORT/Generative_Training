import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_tau_training(save_dir, dataset, model_name):
    tau = 0.125
    save_dir = os.path.join(save_dir, dataset, model_name)

    files = []
    values = []
    for i in range(8):
        name = os.path.join(save_dir, 'data_classif_' + dataset + '-tau' + str((i + 1) * tau) + '.txt')
        files.append(name)
        values.append(np.loadtxt(name))  # [train_loss, train_acc, test_loss, test_acc]

    all_values = np.array(values)

    assert all_values.shape[0] == 8

    max_value = all_values.max(1)

    assert max_value.shape[1] == 4

    x = np.arange(0.125, 1.125, 0.125)

    plt.plot(x, max_value[:, 3])
    plt.savefig(os.path.join(save_dir, 'test_accuracy.png'))

    plt.plot(x, max_value[:, 2])
    plt.savefig(os.path.join(save_dir, 'test_loss.png'))

    plt.plot(x, max_value[:, 1])
    plt.savefig(os.path.join(save_dir, 'train_accuracy.png'))

    plt.plot(x, max_value[:, 0])
    plt.savefig(os.path.join(save_dir, 'train_loss.png'))


plot_tau_training('models','mnist','VAE')
