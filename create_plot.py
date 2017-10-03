import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

data = []
tmp = np.loadtxt("data_classif.txt")
data.append(tmp[:, 1])
data.append(tmp[:, 3])
fig = plt.figure()
ax = plt.subplot()
cm = plt.get_cmap('brg')
ax.plot(data[0], label="Train acc")
ax.plot(data[1], label="Test acc")
legend = ax.legend(loc=4, shadow=True)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and test accuracy on mnist with sample data from VAE")
plt.savefig("acc_cvae_classif.png")
plt.clf()


data = []
tmp = np.loadtxt("data_classif.txt")
data.append(tmp[:, 0])
data.append(tmp[:, 2])
fig = plt.figure()
ax = plt.subplot()
cm = plt.get_cmap('brg')
ax.plot(data[0], label="Train loss")
ax.plot(data[1], label="Test loss")
legend = ax.legend(loc=4, shadow=True)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and test loss on mnist with sample data from VAE")
plt.savefig("loss_cvae_classif.png")
plt.clf()
