# So you think you can generate?

## Abstract

Generative network are trending. A lot of recent papers came out with nice samples of variate kind of images. It's a shame to not use those beautiful samples. What about training another NN with it? Why? Because if it works you prove that your generator fits the distribution of your data and it is à nice transfer strategy. This paper demonstrate that you can train classifier NN based only on samples generate. This framework compare differents generator quality and assure that transfer Can be achieved by this way. Our experiment have show great result on various générative models in mnist , fashion mnsit, cifar10,....

## Experiment

Test Error of the models trained by generative networks

### Tableau de modèles essentiels

| Datasets          | Supervise | VAE  | Conditional VAE | GAN  | Conditional GAN | WGAN | Conditional WGAN |
|-------------------|-----------|------|---------------- |------|---------------- |------|------------------|
| **Mnist**         |  98.92%   |  92% |                 |77.23%|                 |      |                  |
| **Fashon Mnist**  |  87.91%   |  70% |                 |      |                 |      |                  |
|  **Cifar10**      |  59.89%   |      |                 |      |                 |      |                  |


### Tableau secondaire (résultats éventuelles)

| Datasets          | BEGAN  | CGAN | DRAGAN | EBGAN | LSGAN | ACGAN | InfoGAN |
|-------------------|--------|------|--------|-------|-------|-------|---------|
| **Mnist**         |        |      |        |       |       |       |         |
| **Fashon Mnist**  |        |      |        |       |       |       |         |
| **Cifar10**       |        |      |        |       |       |       |         |



# Inspired by Github Repo
https://github.com/wiseodd/generative-models

GAN : [pytorch-generative-model-collections](https://github.com/znxlwm/pytorch-generative-model-collections) <br>
Mnist Classifier : [Pytorch Example](https://github.com/pytorch/examples/tree/master/mnist) <br>
cifar10 (model):[Pytorch Tutorial](https://github.com/pytorch/tutorials)<br>
Fashion Mnist (model) : [fashion-mnist-pytorch](https://github.com/mayurbhangale/fashion-mnist-pytorch/blob/master/CNN_Fashion_MNIST.ipynb)v


# SOTA classifiers
http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html

https://github.com/zalandoresearch/fashion-mnist
[classification_datasets_results](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)<br>
[fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)<br>
