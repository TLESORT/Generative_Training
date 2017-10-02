# So you think you can generate?

## Abstract

Generative network are trending. A lot of recent papers came out with nice samples of variate kind of images. It's a shame to not use those beautiful samples. What about training another NN with it? Why? Because if it works you prove that your generator fits the distribution of your data and it is à nice transfer strategy. This paper demonstrate that you can train classifier NN based only on samples generate. This framework compare differents generator quality and assure that transfer Can be achieved by this way. Our experiment have show great result on various générative models in mnist , fashion mnsit, cifar10,....

## Experiment

Test Error of the models trained by generative networks

### Tableau de modèles essentiels

| Datasets          | VAE  | Conditional VAE | GAN  | Conditional GAN | WGAN | Conditional WGAN |
|-------------------|------|---------------- |------|---------------- |------|------------------|
| **Mnist**         |  95% |                 |      |                 |      |                  |
| **Fashon Mnist**  |  70% |                 |      |                 |      |                  |
|  **Cifar10**      |      |                 |      |                 |      |                  |


### Tableau secondaire (résultats éventuelles)

| Datasets          | BEGAN  | CGAN | DRAGAN | EBGAN | LSGAN | ACGAN | InfoGAN |
|-------------------|--------|------|--------|-------|-------|-------|---------|
| **Mnist**         |        |      |        |       |       |       |         |
| **Fashon Mnist**  |        |      |        |       |       |       |         |
| **Cifar10**       |        |      |        |       |       |       |         |



# Inspired by Github Repo

[pytorch-generative-model-collections](https://github.com/znxlwm/pytorch-generative-model-collections) <br>
[Pytorch Example](https://github.com/pytorch/examples/tree/master/mnist) <br>
