import torch
from multi_vae.MvModels import VAE


def load(view_num=2,
         Network='C',
         hid=256,
         img_size= (1, 32, 32),
         path='./example-model.pt',
         latent_spec = {"disc": [10], "cont": 10},
         shareAE=1):
    """
    Loads a trained model.

    Parameters
    ----------
    path : string
        Path to folder where model is saved. For example
        './models/dataset/'. Note the path MUST end with a '/'
    """
    path_to_model = path
    # Get model
    model = VAE(latent_spec=latent_spec, img_size=img_size, view_num=view_num,
                Network=Network, hidden_dim=hid, shareAE=shareAE)
    model.load_state_dict(torch.load(path_to_model,
                                     map_location=lambda storage, loc: storage))

    return model
