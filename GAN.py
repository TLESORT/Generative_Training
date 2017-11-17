import utils, torch, time
import sort_utils
import numpy as np
from torch.autograd import Variable
from Generative_Model import GenerativeModel


class GAN(GenerativeModel):

    def train_all_classes(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []


        print('training start!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            D_losses = []
            G_losses = []

            epoch_start_time = time.time()
            for tours in range(50000/self.num_examples): #we want to see always as much images
                for x_, y_ in self.data_loader_train:
                    # train discriminator D
                    self.D.zero_grad()

                    batch_size = x_.size()[0]

                    y_real_ = torch.ones(batch_size)
                    y_fake_ = torch.zeros(batch_size)
                    y_label_ = torch.zeros(batch_size, 10)
                    y_label_.scatter_(1, y_.view(batch_size, 1), 1)

                    x_ = x_.view(-1, 1 , 28, 28)
                    x_, y_label_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_label_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
                    D_result, c = self.D(x_, y_label_)#.squeeze()
                    D_real_loss = self.BCELoss(D_result, y_real_)

                    z_ = torch.rand((batch_size, self.z_dim, 1, 1))
                    y_ = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor)

                    z_ = Variable(z_.cuda())
                    G_result = self.G(z_, y_label_)

                    D_result, c = self.D(G_result, y_label_)#.squeeze()
                    D_fake_loss = self.BCELoss(D_result, y_fake_)
                    D_fake_score = D_result.data.mean()

                    D_train_loss = D_real_loss + D_fake_loss

                    D_train_loss.backward()
                    self.D_optimizer.step()

                    D_losses.append(D_train_loss.data[0])

                    # train generator G
                    self.G.zero_grad()

                    z_ = torch.rand((batch_size, self.z_dim, 1, 1))
                    y_ = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor)

                    z_ = Variable(z_.cuda())

                    G_result = self.G(z_, y_label_)
                    D_result,c = self.D(G_result, y_label_)#.squeeze()
                    G_train_loss = self.BCELoss(D_result, y_real_)
                    G_train_loss.backward()
                    self.G_optimizer.step()

                    G_losses.append(G_train_loss.data[0])

                epoch_end_time = time.time()
                per_epoch_ptime = epoch_end_time - epoch_start_time


            print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), self.epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                    torch.mean(torch.FloatTensor(G_losses))))
            self.train_hist['D_loss'].append(torch.mean(torch.FloatTensor(D_losses)))
            self.train_hist['G_loss'].append(torch.mean(torch.FloatTensor(G_losses)))
            self.train_hist['per_epoch_time'].append(per_epoch_ptime)

            self.save()
            self.visualize_results((epoch + 1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        utils.generate_animation(self.result_dir + '/' + self.model_name, self.epoch)
        utils.loss_plot(self.train_hist, self.save_dir, self.model_name)


    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.size_epoch = 1000

        list_classes = sort_utils.get_list_batch(self.data_loader_train)  # list filled all classe sorted by class

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda(self.device)), Variable(
                torch.zeros(self.batch_size, 1).cuda(self.device))
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(
                torch.zeros(self.batch_size, 1))

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for classe in range(10):
            for epoch in range(self.epoch):
                self.G.train()
                epoch_start_time = time.time()
                # for iter, (x_, _) in enumerate(self.data_loader):
                for iter in range(self.size_epoch):
                    x_ = sort_utils.get_batch(list_classes, classe, self.batch_size)
                    # if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    #    break

                    z_ = torch.rand((self.batch_size, self.z_dim))

                    if self.gpu_mode:
                        x_, z_ = Variable(x_.cuda(self.device)), Variable(z_.cuda(self.device))
                    else:
                        x_, z_ = Variable(x_), Variable(z_)

                    # update D network
                    self.D_optimizer.zero_grad()

                    D_real = self.D(x_)
                    D_real_loss = self.BCELoss(D_real, self.y_real_)

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    D_fake_loss = self.BCELoss(D_fake, self.y_fake_)

                    D_loss = D_real_loss + D_fake_loss
                    self.train_hist['D_loss'].append(D_loss.data[0])

                    D_loss.backward()
                    self.D_optimizer.step()

                    # update G network
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = self.BCELoss(D_fake, self.y_real_)
                    self.train_hist['G_loss'].append(G_loss.data[0])

                    G_loss.backward()
                    self.G_optimizer.step()

                    if ((iter + 1) % 100) == 0:
                        print("classe : [%1d] Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                              (classe, (epoch + 1), (iter + 1), self.size_epoch, D_loss.data[0], G_loss.data[0]))

                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                self.visualize_results((epoch + 1), classe)
                self.save_G(classe)
            utils.generate_animation(
                self.result_dir + '/' + 'classe-' + str(classe) + '/' + self.model_name, self.epoch)
            utils.loss_plot(self.train_hist, self.save_dir,self.model_name)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()

