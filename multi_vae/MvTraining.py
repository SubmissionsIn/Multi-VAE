import imageio
import numpy as np
import torch
from torch.nn import functional as F

EPS = 1e-12

class Trainer():
    def __init__(self, model, optimizer, cont_capacity=None,
                 disc_capacity=None, print_loss_every=50, record_loss_every=100,
                 use_cuda=False, view_num=2, DATA='DATA'):
        """
        **Acknowledgments**
        This code is inspired by https://github.com/Schlumberger/joint-vae

        Class to handle training of model.

        Parameters
        ----------
        model : multi-vae.models.VAE instance

        optimizer : torch.optim.Optimizer instance

        cont_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_z).
            Parameters to control the capacity of the continuous latent
            channels. Cannot be None if model.is_continuous is True.

        disc_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_c).
            Parameters to control the capacity of the discrete latent channels.
            Cannot be None if model.is_discrete is True.

        print_loss_every : int
            Frequency with which loss is printed during training.

        record_loss_every : int
            Frequency with which loss is recorded during training.

        use_cuda : bool
            If True moves model and training to GPU.
        """
        self.model = model
        self.optimizer = optimizer
        self.cont_capacity = cont_capacity
        self.disc_capacity = disc_capacity
        self.print_loss_every = print_loss_every
        self.record_loss_every = record_loss_every
        self.use_cuda = use_cuda
        self.view_num = view_num
        self.DATA = DATA
        if self.model.is_continuous and self.cont_capacity is None:
            raise RuntimeError("Model is continuous but cont_capacity not provided.")

        if self.model.is_discrete and self.disc_capacity is None:
            raise RuntimeError("Model is discrete but disc_capacity not provided.")

        if self.use_cuda:
            self.model.cuda()

        # Initialize attributes
        self.num_steps = 0
        self.beta = []
        for i in range(self.view_num):
            self.beta.append(1)
        self.batch_size = None
        self.losses = {'loss': [],
                       'recon_loss': [],
                       'kl_loss': []}
        
        self.loss_r = [[], [], []]  # recon_loss
        self.loss_z = [[], [], []]  # kl_loss z
        self.loss_c = [[], [], []]  # kl_loss c

        self.mean_loss = [[], [], []]

        # Keep track of divergence values for each latent variable
        if self.model.is_continuous:
            self.losses['kl_loss_cont'] = []
            # For every dimension of continuous latent variables
            for i in range(self.model.latent_spec['cont']):
                self.losses['kl_loss_cont_' + str(i)] = []

        if self.model.is_discrete:
            self.losses['kl_loss_disc'] = []
            # For every discrete latent variable
            for i in range(len(self.model.latent_spec['disc'])):
                self.losses['kl_loss_disc_' + str(i)] = []

    def train(self, data_loader, epochs=10, save_training_gif=None):
        """
        Trains the model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader

        epochs : int
            Number of epochs to train the model for.

        save_training_gif : None or tuple (string, Visualizer instance)
            If not None, will use visualizer object to create image of samples
            after every epoch and will save gif of these at location specified
            by string. Note that string should end with '.gif'.
        """
        if save_training_gif is not None:
            training_progress_images = []

        self.batch_size = data_loader.batch_size
        self.model.train()
        c = []
        z = [[], [], []]
        for epoch in range(epochs):
            # print("Epoch:" + str(epoch))
            mean_epoch_loss = self._train_epoch(data_loader)
            for i in range(self.view_num):
                print('Epoch: {}'.format(epoch + 1) + '. Average loss view-' + str(i+1) + ': {:.2f}'.format( self.batch_size * self.model.num_pixels[i] * mean_epoch_loss[i]))
                self.mean_loss[i].append(self.batch_size * self.model.num_pixels[i] * mean_epoch_loss[i])
            show = 0
            if (epoch % 1 == 0) and show == 1:
                batch_size_test = 100000   # max size
                from utils.MvDataloaders import Get_dataloaders
                if self.model.Net == 'C':
                    Train_loader, View_num, N_clusters, _ = Get_dataloaders(batch_size=batch_size_test, DATANAME=self.DATA + '.mat')
                for Batch_idx, Data in enumerate(Train_loader):
                    break
                data = Data[0:-1]
                labels = Data[-1]
                inputs = []
                from torch.autograd import Variable    
                for i in range(View_num):
                    inputs.append(Variable(data[i]))
                    inputs[i] = inputs[i].cuda()
                encodings = self.model.encode(inputs)
                from Nmetrics import test
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=N_clusters, n_init=100)
                x = encodings['disc'][0].cpu().detach().data.numpy()
                y = labels.cpu().detach().data.numpy()
                for i in range(View_num):
                    name = 'cont' + str(i+1)
                    x_c = encodings[name][0].cpu().detach().data.numpy()   # cz
                    cz = kmeans.fit_predict(x_c)
                    z[i].append(test(y, cz))
                p = kmeans.fit_predict(x)
                c.append(test(y, p))
                # yp = x.argmax(1)
                # acc_cmax = test(yp, y)
                # acc_abs = test(p, yp)
                # if acc_abs == 1:
                #     break

            if save_training_gif is not None:
                # Generate batch of images and convert to grid
                viz = save_training_gif[1]
                viz.save_images = False
                img_grid = viz.all_latent_traversals(size=10)
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                # Add image grid to training progress
                training_progress_images.append(img_grid)
        # np.save('./MultiC.npy', c)
        # np.save('./MultiZ.npy', z)
        # np.save('./LossR.npy', self.loss_r)
        # np.save('./LossZ.npy', self.loss_z)
        # np.save('./LossC.npy', self.loss_c)
        np.save('./mean_loss.npy', self.mean_loss)

        if save_training_gif is not None:
            imageio.mimsave(save_training_gif[0], training_progress_images,
                            fps=24)

    def _train_epoch(self, data_loader):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
        """
        epoch_loss = []
        print_every_loss = []
        for i in range(self.view_num):
            epoch_loss.append(0.)
            print_every_loss.append(0.)

        for batch_idx, Data in enumerate(data_loader):
            iter_loss = self._train_iteration(Data[0:-1])
            # print(iter_loss)
            for i in range(self.view_num):
                epoch_loss[i] += iter_loss[i]
                print_every_loss[i] += iter_loss[i]
            # Print loss info every self.print_loss_every iteration
            if batch_idx % self.print_loss_every == 0:
                if batch_idx == 0:
                    mean_loss = print_every_loss
                else:
                    for i in range(self.view_num):
                        mean_loss[i] = print_every_loss[i] / self.print_loss_every
                for i in range(self.view_num):
                    # print('Loss' + str(i+1) + '\t{}/{}\t: {:.3f}'.format(batch_idx * len(Data[i]), len(data_loader.dataset),
                    #                                 self.model.num_pixels * mean_loss[i]))
                    print_every_loss[i] = 0.
        # Return mean epoch loss
        Epoch_loss = []
        for i in range(self.view_num):
            Epoch_loss.append(epoch_loss[i] / len(data_loader.dataset))
        return Epoch_loss

    def _train_iteration(self, data):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            A batch of data. Shape (N, C, H, W)
        """
        self.num_steps += 1
        for i in range(self.view_num):
            if self.use_cuda:
                data[i] = data[i].cuda()

        self.optimizer.zero_grad()
        listout, latent_dist = self.model(data)
        recon_batchs = []
        for i in range(self.view_num):
            recon_batchs.append(listout[i])

        Loss = []

        max_recon_loss = F.binary_cross_entropy(recon_batchs[0].view(-1, self.model.num_pixels[0]),
                                                data[0].view(-1, self.model.num_pixels[0]))
        max_recon_loss *= self.model.num_pixels[0]
        # print(float(max_recon_loss))
        recloss = [max_recon_loss]
        for i in range(self.view_num - 1):
            loss_view = F.binary_cross_entropy(recon_batchs[i+1].view(-1, self.model.num_pixels[i+1]),
                                                data[i+1].view(-1, self.model.num_pixels[i+1]))
            loss_view *= self.model.num_pixels[i+1]
            # print(float(loss_view))
            recloss.append(loss_view)
            if max_recon_loss < loss_view:
                max_recon_loss = loss_view
        # print(self.num_steps)
        if self.num_steps == 1:
            for i in range(self.view_num):
                self.beta[i] = float(recloss[i])/float(max_recon_loss)
            # print(self.beta)

        # print(max_recon_loss)
        # print(latent_dist)
        for i in range(self.view_num):
            countinue_name = 'cont' + str(i+1)
            Loss.append(self._loss_function(data[i], recon_batchs[i], {'cont': latent_dist[countinue_name], 'disc': latent_dist['disc']},
                                            view=i, max_recon_loss=max_recon_loss, beta=self.beta[i]))
        for i in range(self.view_num - 1):
            Loss[i].backward(retain_graph=True)
        Loss[-1].backward()
        self.optimizer.step()

        train_loss = []
        for i in range(self.view_num):
            train_loss.append(Loss[i].item())
        # print(train_loss)
        return train_loss

    def _loss_function(self, data, recon_data, latent_dist, view=0, max_recon_loss=1000, beta=1):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Should have shape (N, C, H, W)

        recon_data : torch.Tensor
            Reconstructed data. Should have shape (N, C, H, W)

        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both containing the parameters
            of the latent distributions as values.
        """
        # Reconstruction loss is pixel wise cross-entropy
        if self.model.Net == 'C':
            recon_loss = F.binary_cross_entropy(recon_data.view(-1, self.model.num_pixels[view]),
                                                data.view(-1, self.model.num_pixels[view]))

        # F.binary_cross_entropy takes mean over pixels, so unnormalise this
        recon_loss *= self.model.num_pixels[view]
        # print(recon_loss, self.model.num_pixels[view])
        # Calculate KL divergences
        kl_cont_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        kl_disc_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        cont_capacity_loss = 0
        disc_capacity_loss = 0

        cont_max, cont_gamma, iters_add_cont_max = \
                self.cont_capacity
        disc_max, disc_gamma, iters_add_disc_max = \
                self.disc_capacity
                
        step = cont_max / iters_add_cont_max


        if self.model.is_continuous:
            # Calculate KL divergence
            mean, logvar = latent_dist['cont']
            kl_cont_loss = self._kl_normal_loss(mean, logvar)
            # Linearly increase capacity of continuous channels
            # Increase continuous capacity without exceeding cont_max
            cont_cap_current = step * self.num_steps
            cont_cap_current = min(cont_cap_current, cont_max)
            # Calculate continuous capacity loss
            if self.DATA in ['Object-Digit-Product']:
                # cont_gamma = cont_gamma * recon_loss/max_recon_loss
                cont_gamma = cont_gamma * beta
            cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_cont_loss)
        if self.model.is_discrete:
            # Calculate KL divergence
            kl_disc_loss = self._kl_multiple_discrete_loss(latent_dist['disc'])
            # Linearly increase capacity of discrete channels
            # Increase discrete capacity without exceeding disc_max or theoretical
            disc_cap_current = step * self.num_steps
            disc_cap_current = min(disc_cap_current, disc_max)
            # Calculate discrete capacity loss
            if self.DATA in ['Object-Digit-Product']:
                # disc_gamma = disc_gamma * recon_loss/max_recon_loss
                disc_gamma = disc_gamma * beta
            disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_disc_loss)
        # Calculate total kl value to record it
        kl_loss = kl_cont_loss + kl_disc_loss
        # Calculate total loss
        total_loss = recon_loss + cont_capacity_loss + disc_capacity_loss
        # print(self.num_steps, recon_loss, cont_capacity_loss+disc_capacity_loss, kl_loss)

        # Record losses
        # if self.model.training and self.num_steps % self.record_loss_every == 0:
        #     self.losses['recon_loss'].append(recon_loss.item())
        #     self.losses['kl_loss'].append(kl_loss.item())
        #     self.losses['loss'].append(total_loss.item())
        #     print(recon_loss.data, kl_cont_loss.data, kl_disc_loss.data)

        # self.loss_r[view].append(recon_loss.item())
        # self.loss_z[view].append(kl_cont_loss.item())
        # self.loss_c[view].append(kl_disc_loss.item())

        # To avoid large losses normalise by number of pixels
        return total_loss / self.model.num_pixels[view]

    def _kl_normal_loss(self, mean, logvar):
        """
        Calculates the KL divergence between a normal distribution with
        diagonal covariance and a unit normal distribution.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        # Calculate KL divergence
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        # Mean KL divergence across batch for each latent variable
        kl_means = torch.mean(kl_values, dim=0)
        # KL loss is sum of mean KL of each latent variable
        kl_loss = torch.sum(kl_means)

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['kl_loss_cont'].append(kl_loss.item())
            for i in range(self.model.latent_spec['cont']):
                self.losses['kl_loss_cont_' + str(i)].append(kl_means[i].item())

        return kl_loss

    def _kl_multiple_discrete_loss(self, alphas):
        """
        Calculates the KL divergence between a set of categorical distributions
        and a set of uniform categorical distributions.

        Parameters
        ----------
        alphas : list
            List of the alpha parameters of a categorical (or gumbel-softmax)
            distribution. For example, if the categorical atent distribution of
            the model has dimensions [2, 5, 10] then alphas will contain 3
            torch.Tensor instances with the parameters for each of
            the distributions. Each of these will have shape (N, D).
        """
        # Calculate kl losses for each discrete latent
        kl_losses = [self._kl_discrete_loss(alpha) for alpha in alphas]

        # Total loss is sum of kl loss for each discrete latent
        kl_loss = torch.sum(torch.cat(kl_losses))

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['kl_loss_disc'].append(kl_loss.item())
            for i in range(len(alphas)):
                self.losses['kl_loss_disc_' + str(i)].append(kl_losses[i].item())

        return kl_loss

    def _kl_discrete_loss(self, alpha):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D)
        """
        disc_dim = int(alpha.size()[-1])
        log_dim = torch.Tensor([np.log(disc_dim)])
        if self.use_cuda:
            log_dim = log_dim.cuda()
        # Calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        # Take mean of negative entropy across batch
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)
        # KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy
        return kl_loss
