# So you think you can generate?


## Overlead
https://www.overleaf.com/11429932yftxswvttwbs

## Instruction

```
python main.py --dataset mnist --gan_type <TYPE> --epoch 25 --batch_size 64
```

## Abstract

Generative network are trending. A lot of recent papers came out with nice samples of variate kind of images. But those generator are difficult to assess. A good generator should generate data which contain meaningful and variate information and that fit the distribution of the training set. This paper present a new method to assess a generator. This method is based on a metric which measure the information contains in the data produce by the generator. We used the train set of labeled dataset $D$ to train a generator. We transfer the information captured by the generator by generating a new labeled dataset $\hat{D}$ which should fit $D$ distribution. $\hat{D}$ is then use to train a classifier in a supervised fashion. The classifier is afterwards tested on the test set of $D$ in order to measure if the generator successfully produce images that fits the distribution of the dataset. We called this method "assessment by transfer measurement". Our experiment compare the result of different generator from the VAE and GAN framework with the dataset mnist, fashion mnist and cifar10.

## Experiment

Test Error of the models trained by generative networks

### Tableau de modèles essentiels

| Datasets          | Supervise | VAE  | Conditional VAE | GAN  | Conditional GAN | WGAN | Conditional WGAN |
|-------------------|-----------|------|---------------- |------|---------------- |------|------------------|
| **Mnist**         |  98.92%   |  92% |     98.47%      |77.23%|                 |      |                  |
| **Fashon Mnist**  |  87.91%   |  70% |     77.84%      |      |                 |      |                  |
|  **Cifar10**      |  59.89%   |      |     30.39%      |      |                 |      |                  |


### Tableau secondaire (résultats éventuelles)

| Datasets          | BEGAN  | CGAN | DRAGAN | EBGAN | LSGAN | ACGAN | InfoGAN |
|-------------------|--------|------|--------|-------|-------|-------|---------|
| **Mnist**         |        |      |        |       |       |       |         |
| **Fashon Mnist**  |        |      |        |       |       |       |         |
| **Cifar10**       |        |      |        |       |       |       |         |



# Inspired by Github Repo

GAN : [pytorch-generative-model-collections](https://github.com/znxlwm/pytorch-generative-model-collections) <br>
Mnist Classifier : [Pytorch Example](https://github.com/pytorch/examples/tree/master/mnist) <br>
cifar10 (model):[Pytorch Tutorial](https://github.com/pytorch/tutorials)<br>
Fashion Mnist (model) : [fashion-mnist-pytorch](https://github.com/mayurbhangale/fashion-mnist-pytorch/blob/master/CNN_Fashion_MNIST.ipynb)v


# SOTA classifiers
[classification_datasets_results](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)<br>
[fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)<br>
