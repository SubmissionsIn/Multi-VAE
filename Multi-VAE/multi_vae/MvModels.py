import torch
from torch import nn
from torch.nn import functional as F
EPS = 1e-12


class VAE(nn.Module):
    def __init__(self, img_size, latent_spec, temperature=.67, 
                 use_cuda=False, view_num=2, Network='C',
                 hidden_dim=256, shareAE=1):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32)

        latent_spec : dict
            Specifies latent distribution.

        temperature : float
            Temperature for gumbel softmax distribution.

        use_cuda : bool
            If True moves model to GPU
        """
        super(VAE, self).__init__()
        self.use_cuda = use_cuda

        # Parameters
        self.img_size = img_size
        self.is_continuous = 'cont' in latent_spec
        self.is_discrete = 'disc' in latent_spec
        self.latent_spec = latent_spec

        self.view_num = view_num
        self.MvLatent_spec = {'disc': latent_spec['disc']}
        for i in range(self.view_num):
            continue_name = 'cont' + str(i + 1)
            self.MvLatent_spec[continue_name] = latent_spec['cont']
        
        self.Net = Network
        self.share = shareAE
        if self.Net == 'C':
            self.num_pixels = []
            for i in range(self.view_num):
                self.num_pixels.append(img_size[1] * img_size[2])
            
        self.temperature = temperature
        self.hidden_dim = hidden_dim  # Hidden dimension of linear layer
        self.reshape = (64, 4, 4)  # Shape required to start transpose convs

        # Calculate dimensions of latent distribution
        self.latent_cont_dim = 0
        self.latent_disc_dim = 0
        self.num_disc_latents = 0
        if self.is_continuous:
            self.latent_cont_dim = self.latent_spec['cont']
        if self.is_discrete:
            self.latent_disc_dim += sum([dim for dim in self.latent_spec['disc']])
            self.num_disc_latents = len(self.latent_spec['disc'])
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim

        if self.Net == 'C':
            Mv_img_to_features = []
            Mv_features_to_hidden = []
            Mv_latent_to_features = []
            Mv_features_to_img = []
            # Define encoder layers
            for i in range(self.view_num):
                # Intial layer
                encoder_layers = [
                    nn.Conv2d(self.img_size[0], 32, (4, 4), stride=2, padding=1),
                    nn.ReLU()
                ]
                # Add additional layer if (64, 64) images
                if self.img_size[1:] == (64, 64):
                    encoder_layers += [
                        nn.Conv2d(32, 32, (4, 4), stride=2, padding=1),
                        nn.ReLU()
                    ]
                elif self.img_size[1:] == (32, 32):
                    # (32, 32) images are supported but do not require an extra layer
                    pass
                else:
                    raise RuntimeError(
                        "{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(
                            img_size))
                # Add final layers
                encoder_layers += [
                    nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
                    nn.ReLU()
                ]
                # Define encoder
                Mv_img_to_features.append(nn.Sequential(*encoder_layers))
                # Map encoded features into a hidden vector which will be used to
                # encode parameters of the latent distribution
                features_to_hidden = nn.Sequential(
                    nn.Linear(64 * 4 * 4, self.hidden_dim),
                    nn.ReLU()
                    # nn.Sigmoid()
                )
                Mv_features_to_hidden.append(features_to_hidden)

            if self.share:
                self.Mv_img_to_features = nn.ModuleList([Mv_img_to_features[0]])
                self.Mv_features_to_hidden = nn.ModuleList([Mv_features_to_hidden[0]])
            else:
                self.Mv_img_to_features = nn.ModuleList(Mv_img_to_features)
                self.Mv_features_to_hidden = nn.ModuleList(Mv_features_to_hidden)
            # Encode parameters of latent distribution
            means = []
            log_vars = []
            if self.is_continuous:
                for i in range(self.view_num):
                    means.append(nn.Linear(self.hidden_dim, self.latent_cont_dim))
                    log_vars.append(nn.Linear(self.hidden_dim, self.latent_cont_dim))
            self.means = nn.ModuleList(means)
            self.log_vars = nn.ModuleList(log_vars)
            if self.is_discrete:
                # Linear layer for each of the categorical distributions
                fc_alphas = []
                for disc_dim in self.latent_spec['disc']:
                    fc_alphas.append(nn.Linear(self.hidden_dim * self.view_num, disc_dim))
                self.fc_alphas = nn.ModuleList(fc_alphas)

            for i in range(self.view_num):
                # Map latent samples to features to be used by generative model
                latent_to_features = nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 64 * 4 * 4),
                    nn.ReLU()
                )
                Mv_latent_to_features.append(latent_to_features)
                # Define decoder
                decoder_layers = []
                # Additional decoding layer for (64, 64) images
                if self.img_size[1:] == (64, 64):
                    decoder_layers += [
                        nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=1),
                        nn.ReLU()
                    ]

                decoder_layers += [
                    nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, self.img_size[0], (4, 4), stride=2, padding=1),
                    nn.Sigmoid()
                    # nn.Tanh()
                ]
                # Define decoder
                Mv_features_to_img.append(nn.Sequential(*decoder_layers))

            if self.share:
                self.Mv_latent_to_features = nn.ModuleList([Mv_latent_to_features[0]])
                self.Mv_features_to_img = nn.ModuleList([Mv_features_to_img[0]])
            else:
                self.Mv_latent_to_features = nn.ModuleList(Mv_latent_to_features)
                self.Mv_features_to_img = nn.ModuleList(Mv_features_to_img)

    def encode(self, X):
        """
        Encodes an image into parameters of a latent distribution defined in
        self.latent_spec.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data, shape (N, C, H, W)
        """
        batch_size = X[0].size()[0]
        features = []
        hiddens = []
        for i in range(self.view_num):
            # Encode image to hidden features
            if self.share:
                net_num = 0
            else:
                net_num = i
            # print(X[i].shape)
            features.append(self.Mv_img_to_features[net_num](X[i]))
            hiddens.append(self.Mv_features_to_hidden[net_num](features[i].view(batch_size, -1)))

        fusion = torch.cat(hiddens, dim=1)
        # print(hidden.shape)
        # print(fusion.shape)
        # Output parameters of latent distribution from hidden representation
        latent_dist = {}
        if self.is_continuous:
            for i in range(self.view_num):
                continue_name = 'cont' + str(i+1)
                latent_dist[continue_name] = [self.means[i](hiddens[i]), self.log_vars[i](hiddens[i])]
        if self.is_discrete:
            latent_dist['disc'] = []
            for fc_alpha in self.fc_alphas:
                latent_dist['disc'].append(F.softmax(fc_alpha(fusion), dim=1))

        return latent_dist

    def reparameterize(self, latent_dist):
        """
        Samples from latent distribution using the reparameterization trick.

        Parameters
        ----------
        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both, containing the parameters
            of the latent distributions as torch.Tensor instances.
        """
        latent_sample = []
        if self.is_continuous:
            for i in range(self.view_num):
                countinus_name = 'cont' + str(i+1)
                mean, logvar = latent_dist[countinus_name]
                cont_sample = self.sample_normal(mean, logvar)
                latent_sample.append(cont_sample)

        if self.is_discrete:
            for alpha in latent_dist['disc']:
                disc_sample = self.sample_gumbel_softmax(alpha)
                latent_sample.append(disc_sample)

        # Concatenate continuous and discrete samples into one large sample
        return torch.cat(latent_sample, dim=1)

    def sample_normal(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros(std.size()).normal_()
            if self.use_cuda:
                eps = eps.cuda()
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def sample_gumbel_softmax(self, alpha):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """
        if self.training:
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size())
            if self.use_cuda:
                unif = unif.cuda()
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
            if self.use_cuda:
                one_hot_samples = one_hot_samples.cuda()
            return one_hot_samples

    def decode(self, latent_samples):
        """
        Decodes sample from latent distribution into an image.

        Parameters
        ----------
        latent_sample : torch.Tensor
            Sample from latent distribution.
        """
        features_to_img = []
        for i in range(self.view_num):
            if self.share:
                net_num = 0
            else:
                net_num = i
            feature = self.Mv_latent_to_features[net_num](latent_samples[i])
            if self.Net == 'C':
                features_to_img.append(self.Mv_features_to_img[net_num](feature.view(-1, *self.reshape)))
        return features_to_img[:]

    def forward(self, X):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (N, C, H, W)
        """
        latent_dist = self.encode(X)
        # print(latent_dist)
        latent_sample = self.reparameterize(latent_dist)
        # print(latent_sample.shape)
        split_list = []
        for i in range(self.view_num):
            split_list.append(self.latent_spec['cont'])
        split_list.append(self.latent_spec['disc'][0])
        cont_des_list = latent_sample.split(split_list, dim=1)
        decode_list = []
        for i in range(self.view_num):
            decode_list.append(torch.cat([cont_des_list[i], cont_des_list[-1]], dim=1))
        out_list = self.decode(decode_list)
        return out_list, latent_dist
