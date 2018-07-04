import utils, torch, time, os, pickle
import numpy as np
from torch import autograd
from torch.autograd import grad
from torch.autograd import Variable

from Data.load_dataset import get_iter_dataset
from Generative_Models.Generative_Model import GenerativeModel




class WGAN_GP(GenerativeModel):
    def __init__(self, args):

        super(WGAN_GP, self).__init__(args)

        self.lambda_ = 0.25

        # Loss weight for gradient penalty
        self.lambda_gp = 10
        self.cuda = True
        self.c = 0.01  # clipping value
        self.n_critic = 5  # the number of iterations of the critic per generator iteration

        self.y_real_ = torch.FloatTensor([1])
        self.y_fake_ = self.y_real_ * -1


        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(self.device), self.y_fake_.cuda(self.device)





    def compute_gradient_penalty(self, D, real_samples, fake_samples):

        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1, 1))

        if self.gpu_mode:
            alpha=alpha.cuda()

        # Get random interpolation between real and fake samples
        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)

        interpolates = Variable(interpolates, requires_grad=True)

        d_interpolates = D(interpolates)

        fake = Variable(torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        if self.gpu_mode:
            fake=fake.cuda(self.device)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                  grad_outputs=fake, create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients + 1e-16

        gradient_penalty = self.lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        if gradient_penalty.data.mean() != gradient_penalty.data.mean():
            print("GP")
            print("NaN attack !!")


        return gradient_penalty

    def train(self):
        self.G.apply(self.G.weights_init)
        print(' training start!! (no conditional)')
        start_time = time.time()
        for classe in range(10):
            self.train_hist = {}
            self.train_hist['D_loss'] = []
            self.train_hist['G_loss'] = []
            self.train_hist['per_epoch_time'] = []
            self.train_hist['total_time'] = []
            # self.G.apply(self.G.weights_init) does not work for instance
            self.G.train()

            data_loader_train = get_iter_dataset(self.dataset_train, self.list_class_train, self.batch_size, classe)

            print("Classe: " + str(classe))
            for epoch in range(self.epoch):

                epoch_start_time = time.time()

                for iter, (x_, t_) in enumerate(data_loader_train):

                    if iter == data_loader_train.dataset.__len__() // self.batch_size:
                        break

                    z_ = torch.rand((self.batch_size, self.z_dim))

                    if self.gpu_mode:
                        x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
                    else:
                        x_, z_ = Variable(x_), Variable(z_)

                    # update D network
                    self.D_optimizer.zero_grad()

                    D_real = self.D(x_)
                    D_real_loss = -torch.mean(D_real)

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    D_fake_loss = torch.mean(D_fake)

                    # gradient penalty
                    if self.gpu_mode:
                        alpha = torch.rand(x_.size()).cuda()
                    else:
                        alpha = torch.rand(x_.size())

                    x_hat = Variable(alpha * x_.data + (1 - alpha) * G_.data, requires_grad=True)

                    pred_hat = self.D(x_hat)
                    if self.gpu_mode:
                        gradients = \
                        grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
                    else:
                        gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                         create_graph=True, retain_graph=True, only_inputs=True)[0]

                    gradient_penalty = self.lambda_ * (
                    (gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

                    D_loss = D_real_loss + D_fake_loss + gradient_penalty

                    D_loss.backward()
                    self.D_optimizer.step()

                    if ((iter + 1) % self.n_critic) == 0:
                        # update G network
                        self.G_optimizer.zero_grad()

                        G_ = self.G(z_)
                        D_fake = self.D(G_)
                        G_loss = -torch.mean(D_fake)
                        self.train_hist['G_loss'].append(G_loss.data[0])

                        G_loss.backward()
                        self.G_optimizer.step()

                        self.train_hist['D_loss'].append(D_loss.data[0])

                    if ((iter + 1) % 100) == 0:
                        print("classe : [%1d] Epoch: [%2d] [%4d/%4d] G_loss: %.8f, D_loss: %.8f" %
                              (classe, (epoch + 1), (iter + 1), self.size_epoch, G_loss.data[0], D_loss.data[0]))
                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                self.visualize_results((epoch + 1), classe)
            self.save_G(classe)

            result_dir = self.result_dir + '/' + 'classe-' + str(classe)
            utils.generate_animation(result_dir + '/' + self.model_name, epoch+1)
            utils.loss_plot(self.train_hist, result_dir, self.model_name)

            np.savetxt(
                os.path.join(result_dir, 'wgan_training_' + self.dataset + '.txt'),
                np.transpose([self.train_hist['G_loss']]))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    def run_batch(self, x_, iter):

        if x_.data.mean() != x_.data.mean():
            print("Data")
            print("NaN attack !!")

        ############################
        # (1) Update D network
        ###########################
        for p in self.D.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update


        real_data = x_.view(-1, self.input_size, self.size, self.size)
        if self.gpu_mode:
            real_data = real_data.cuda(self.device)

        self.D.zero_grad()

        # train with real
        D_real = self.D(real_data)
        D_real = D_real.mean()
        # print D_real
        D_real.backward(self.y_fake_)

        # train with fake
        noise = torch.randn(x_.size(0), self.z_dim, 1, 1)
        if self.gpu_mode:
            noise = noise.cuda(self.device)
        noisev = Variable(noise, volatile=True)  # totally freeze netG

        fake = Variable(self.G(noisev).data)
        D_fake = self.D(fake)
        D_fake = D_fake.mean()
        D_fake.backward(self.y_real_)


        if D_fake.data.mean() != D_fake.data.mean():
            print("FD")
            print("NaN attack !!")

        # train with gradient penalty
        gradient_penalty = self.compute_gradient_penalty(self.D, real_data.data, fake.data)



        gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        self.D_optimizer.step()

        ############################
        # (2) Update G network
        ###########################
        if ((iter + 1) % self.n_critic) == 0:
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation
            self.G.zero_grad()

            noise = torch.randn(x_.size(0), self.z_dim, 1, 1)
            if self.gpu_mode:
                noise = noise.cuda(self.device)
            noisev = autograd.Variable(noise)
            fake = self.G(noisev)
            G = self.D(fake)
            G = G.mean()
            G.backward(self.y_fake_)
            G_cost = -G
            self.G_optimizer.step()


