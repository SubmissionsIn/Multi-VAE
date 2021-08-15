import numpy as np
import torch
from viz.latent_traversals import LatentTraverser
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image


class Visualizer():
    def __init__(self, model, view_num=2, show_view=1):
        """
        Visualizer is used to generate images of samples, reconstructions,
        latent traversals and so on of the trained model.

        Parameters
        ----------
        model : multi-vae.models.VAE instance
        """
        self.view_num = view_num
        self.show_view = show_view - 1
        self.model = model
        self.latent_traverser = LatentTraverser(self.model.latent_spec)
        self.save_images = False   # If false, each method returns a tensor
                                   # instead of saving image.

    def reconstructions(self, data, size=(8, 8), filename='recon.png'):
        """
        Generates reconstructions of data through the model.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        size : tuple of ints
            Size of grid on which reconstructions will be plotted. The number
            of rows should be even, so that upper half contains true data and
            bottom half contains reconstructions
        """
        # Plot reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        # Pass data through VAE to obtain reconstruction

        # input_data = Variable(data, volatile=True)
        input_datas = []
        with torch.no_grad():
            for i in range(self.view_num):
                input_data = Variable(data[i])
                if self.model.use_cuda:
                    input_datas.append(input_data.cuda())
                else:
                    input_datas.append(input_data)
        view = self.show_view
        recon_datas, prior_samples = self.model(input_datas)
        print(prior_samples)

        # prior_sample = self.latent_traverser.traverse_grid(size=(5, 10),
        #                                                       cont_idx=None,
        #                                                       cont_axis=None,
        #                                                       disc_idx=0,
        #                                                       disc_axis=0)
        # print(prior_sample.shape)
        # print(prior_samples['disc'][0].shape)
        # for i in range(50):
        #     for j in range(10):
        #         prior_sample[i, j+10] = prior_samples['disc'][0][i, j]
        # generateds = self._decode_latents([prior_sample, prior_sample, prior_sample])
        # if self.save_images:
        #     save_image(generateds[view].data, 'cluster.png', nrow=size[1])
        # else:
        #     return make_grid(generateds[view].data, nrow=size[1])

        self.model.train()
        # Upper half of plot will contain data, bottom half will contain
        # reconstructions
        num_images = size[0] * size[1] / 2
        num_images = np.int(num_images)

        originals = input_datas[view][:num_images].cpu()
        reconstructions = recon_datas[view].view(-1, *self.model.img_size)[:num_images].cpu()
        # If there are fewer examples given than spaces available in grid,
        # augment with blank images
        num_examples = originals.size()[0]
        if num_images > num_examples:
            blank_images = torch.zeros((num_images - num_examples,) + originals.size()[1:])
            originals = torch.cat([originals, blank_images])
            reconstructions = torch.cat([reconstructions, blank_images])

        # Concatenate images and reconstructions
        comparison = torch.cat([originals, reconstructions])

        if self.save_images:
            save_image(comparison.data, filename, nrow=size[0])
        else:
            return make_grid(comparison.data, nrow=size[0])

    def samples(self, size=(10, 10), filename='samples.png'):
        """
        Generates samples from learned distribution by sampling prior and
        decoding.

        size : tuple of ints
        """
        # Get prior samples from latent distribution
        cached_sample_prior = self.latent_traverser.sample_prior
        self.latent_traverser.sample_prior = True
        prior_samples_1 = self.latent_traverser.traverse_grid(size=size,
                                                              cont_idx=None,
                                                              cont_axis=None,
                                                              disc_idx=0,
                                                              disc_axis=0)
        self.latent_traverser.sample_prior = cached_sample_prior
        # Map samples through decoder
        print("------------------------------")
        print(prior_samples_1.shape)
        prior_zeros = torch.zeros(prior_samples_1.shape)
        # prior_zeros = torch.ones(prior_samples_1.shape)
        # print(prior_zeros.shape)
        # print(prior_samples_1[:, 0:10].shape)
        # prior_samples_1 = torch.cat([prior_samples_1[:, 0:10], prior_zeros[:, 10:20]], dim=1)
        print(prior_samples_1)
        # print(prior_samples_1[0:10])
        print("------------------------------")
        prior_samples = []
        for i in range(self.view_num):
            prior_samples.append(prior_samples_1)
        generateds = self._decode_latents(prior_samples)

        if self.save_images:
            save_image(generateds[self.show_view].data, filename, nrow=size[1])
        else:
            return make_grid(generateds[self.show_view].data, nrow=size[1])

    def latent_traversal_line(self, cont_idx=None, disc_idx=None, size=8,
                              filename='traversal_line.png'):
        """
        Generates an image traversal through a latent dimension.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_line for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                             disc_idx=disc_idx,
                                                             size=size)

        # Map samples through decoder
        x = torch.cat([latent_samples, latent_samples, latent_samples], dim=0)
        for i in range(10):
            x[3 * i] = latent_samples[i]
            x[3 * i + 1] = x[3 * i]
            x[3 * i + 2] = x[3 * i]
        print(x)
        prior_samples = []
        for i in range(self.view_num):
            prior_samples.append(x)
        # print(prior_samples)
        generateds = self._decode_latents(prior_samples)

        if self.save_images:
            save_image(generateds[self.show_view].data, filename, nrow=3)
        else:
            return make_grid(generateds[self.show_view].data, nrow=3)

    def latent_traversal_grid(self, cont_idx=None, cont_axis=None,
                              disc_idx=None, disc_axis=None, size=(5, 5),
                              filename='traversal_grid.png'):
        """
        Generates a grid of image traversals through two latent dimensions.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_grid for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_grid(cont_idx=cont_idx,
                                                             cont_axis=cont_axis,
                                                             disc_idx=disc_idx,
                                                             disc_axis=disc_axis,
                                                             size=size)

        # Map samples through decoder
        prior_samples = []
        for i in range(self.view_num):
            prior_samples.append(latent_samples)
        generateds = self._decode_latents(prior_samples)

        if self.save_images:
            save_image(generateds[self.show_view].data, filename, nrow=size[1])
        else:
            return make_grid(generateds[self.show_view].data, nrow=size[1])

    def all_latent_traversals(self, size=10, filename='all_traversals.png'):
        """
        Traverses all latent dimensions one by one and plots a grid of images
        where each row corresponds to a latent traversal of one latent
        dimension.

        Parameters
        ----------
        size : int
            Number of samples for each latent traversal.
        """
        latent_samples = []

        # Perform line traversal of every continuous and discrete latent
        for cont_idx in range(self.model.latent_cont_dim):
            latent_samples.append(self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                                      disc_idx=None,
                                                                      size=size))

        for disc_idx in range(self.model.num_disc_latents):
            latent_samples.append(self.latent_traverser.traverse_line(cont_idx=None,
                                                                      disc_idx=disc_idx,
                                                                      size=size))

        # Decode samples
        samples = []
        for i in range(self.view_num):
            samples.append(torch.cat(latent_samples, dim=0))
        generateds = self._decode_latents(samples)

        if self.save_images:
            save_image(generateds[self.show_view].data, filename, nrow=size)
        else:
            return make_grid(generateds[self.show_view].data, nrow=size)

    def _decode_latents(self, latent_sample):
        """
        Decodes latent samples into images.

        Parameters
        ----------
        latent_samples : torch.autograd.Variable
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        latent_samples = []
        for i in range(self.view_num):
            latent_samples.append(Variable(latent_sample[i]))
            if self.model.use_cuda:
                latent_samples[i] = latent_samples[i].cuda()

        return self.model.decode(latent_samples)


def reorder_img(orig_img, reorder, by_row=True, img_size=(3, 32, 32), padding=2):
    """
    Reorders rows or columns of an image grid.

    Parameters
    ----------
    orig_img : torch.Tensor
        Original image. Shape (channels, width, height)

    reorder : list of ints
        List corresponding to desired permutation of rows or columns

    by_row : bool
        If True reorders rows, otherwise reorders columns

    img_size : tuple of ints
        Image size following pytorch convention

    padding : int
        Number of pixels used to pad in torchvision.utils.make_grid
    """
    reordered_img = torch.zeros(orig_img.size())
    _, height, width = img_size

    for new_idx, old_idx in enumerate(reorder):
        if by_row:
            start_pix_new = new_idx * (padding + height) + padding
            start_pix_old = old_idx * (padding + height) + padding
            reordered_img[:, start_pix_new:start_pix_new + height, :] = orig_img[:, start_pix_old:start_pix_old + height, :]
        else:
            start_pix_new = new_idx * (padding + width) + padding
            start_pix_old = old_idx * (padding + width) + padding
            reordered_img[:, :, start_pix_new:start_pix_new + width] = orig_img[:, :, start_pix_old:start_pix_old + width]

    return reordered_img