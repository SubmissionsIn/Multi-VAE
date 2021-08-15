import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import scipy.io as scio


def Get_dataloaders(batch_size=128, path_to_data='./utils/DATA/', DATANAME='MNIST-Sobel.mat'):
    """dataloader with (32, 32) images."""
    DATA = scio.loadmat(path_to_data + DATANAME)
    view = len(DATA)-3-1
    X1 = DATA['X1']
    X2 = DATA['X2']
    print(X1.shape)
    print(X2.shape)
    if view == 3:
        X3 = DATA['X3']
        print(X3.shape)
    y = DATA['Y']
    size = y.shape[1]
    print(y.shape[1])
    cluster = np.unique(y)
    # print(cluster)
    print('Cluster K:' + str(len(cluster)))
    x1 = torch.from_numpy(X1).float()
    X1 = []
    x2 = torch.from_numpy(X2).float()
    X2 = []
    if view == 3:
        x3 = torch.from_numpy(X3).float()
        X3 = []
    y = torch.from_numpy(y[0])
    if view == 2:
        X = TensorDataset(x1, x2, y)
    if view == 3:
        X = TensorDataset(x1, x2, x3, y)
    train_loader = DataLoader(X, batch_size=batch_size, shuffle=True)
    return train_loader, view, len(cluster), size
