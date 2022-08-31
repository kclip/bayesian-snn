import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
from neurodata.load_data import create_dataloader
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from data.mnist import EncodedMNIST


def make_moon_dataset(n_samples, T, noise, n_neuron_per_dim, res=100):
    '''
    Generates points from the Two Moons dataset,
    rescale them in [0, 1] x [0, 1] and encodes them using population coding
    '''

    from sklearn import datasets
    data = datasets.make_moons(n_samples=n_samples, noise=noise)

    c_intervals = res / max(n_neuron_per_dim - 1, 1)
    c = np.arange(0, res + c_intervals, c_intervals)

    data[0][:, 0] = (data[0][:, 0] - np.min(data[0][:, 0])) \
                    / (np.max(data[0][:, 0]) - np.min(data[0][:, 0]))
    data[0][:, 1] = (data[0][:, 1] - np.min(data[0][:, 1])) \
                    / (np.max(data[0][:, 1]) - np.min(data[0][:, 1]))

    binary_inputs = torch.zeros([len(data[0]), T, 2 * n_neuron_per_dim])
    binary_outputs = torch.zeros([len(data[0]), 1])

    for i, sample in enumerate(data[0]):
        rates_0 \
            = np.array([0.5 + np.cos(max(-np.pi, 
                                         min(np.pi,
                                             np.pi * (sample[0] * res - c[k]) 
                                             / c_intervals))) / 2
                        for k in range(n_neuron_per_dim)]).T
        binary_inputs[i, :, :n_neuron_per_dim] \
            = torch.bernoulli(torch.tensor(rates_0).unsqueeze(0).repeat(T, 1))

        rates_1 \
            = np.array([0.5 + np.cos(max(-np.pi,
                                         min(np.pi, 
                                             np.pi * (sample[1] * res - c[k])
                                             / c_intervals))) / 2
                        for k in range(n_neuron_per_dim)]).T
        binary_inputs[i, :, n_neuron_per_dim:] \
            = torch.bernoulli(torch.tensor(rates_1).unsqueeze(0).repeat(T, 1))

        binary_outputs[i, :] = data[1][i]

    return binary_inputs.transpose(1, 2), binary_outputs, \
           torch.FloatTensor(data[0]), torch.FloatTensor(data[1])


def make_moon_test(n_samples_per_dim, T, n_neuron_per_dim, res=100):
    '''
    Generates a grid of equally spaced points in [0, 1] x [0, 1] and encodes 
    them as binary signals using population coding
    '''

    n_samples = n_samples_per_dim ** 2

    c_intervals = res / max(n_neuron_per_dim - 1, 1)
    c = np.arange(0, res + c_intervals, c_intervals)

    binary_inputs = torch.zeros([n_samples, T, 2 * n_neuron_per_dim])

    y, x = np.meshgrid(np.arange(n_samples_per_dim),
                       np.arange(n_samples_per_dim))
    x = (x / n_samples_per_dim).flatten()
    y = (y / n_samples_per_dim).flatten()

    for i in range(n_samples):
        rates_0 \
            = np.array([0.5 + np.cos(max(-np.pi, 
                                         min(np.pi, np.pi * (x[i] * res - c[k]) 
                                             / c_intervals))) / 2
                        for k in range(n_neuron_per_dim)]).T
        binary_inputs[i, :, :n_neuron_per_dim] \
            = torch.bernoulli(torch.tensor(rates_0).unsqueeze(0).repeat(T, 1))

        rates_1 \
            = np.array([0.5 + np.cos(max(-np.pi,
                                         min(np.pi, np.pi * (y[i] * res - c[k])
                                             / c_intervals))) / 2
                        for k in range(n_neuron_per_dim)]).T
        binary_inputs[i, :, n_neuron_per_dim:] \
            = torch.bernoulli(torch.tensor(rates_1).unsqueeze(0).repeat(T, 1))

    return binary_inputs.transpose(1, 2), \
           torch.FloatTensor(x), torch.FloatTensor(y)


class CustomDataset(torch.utils.data.Dataset):
    '''
    Wrapper to create dataloaders from the synthetic datasets
    '''

    def __init__(self, data, target):
        self.data = data
        self.target = target
        super(CustomDataset, self).__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key], self.target[key]


def encode_delta(x, deltas):
    if len(deltas) == 0:
        deltas = [deltas]
    x_delta = np.zeros(x.shape + (len(deltas), 2))
    for i in range(1, x.shape[1]):
        for j, delta in enumerate(deltas):
            x_delta[:, i, j, 0] = (x[:, i] - x[:, i-1]) > delta
            x_delta[:, i, j, 1] = (x[:, i] - x[:, i-1]) < -delta
    x_delta = torch.Tensor(x_delta).view(x.shape + (2 * len(deltas),))
    return x_delta.transpose(1, 2)


def make_mnist_dataloader(digits, batch_size, T):
    mnist_train = EncodedMNIST('./data', train=True, download=True,
                        transform=torchvision.transforms.ToTensor(),
                        classes=digits, T=T)
    train_dl = DataLoader(mnist_train,
                          shuffle=True,
                          batch_size=batch_size)
    mnist_test = EncodedMNIST('./data', train=False, download=True,
                       transform=ToTensor(),
                       classes=digits, T=T)
    test_dl = DataLoader(mnist_test,
                         shuffle=False,
                         batch_size=batch_size)

    return train_dl, test_dl


def make_dvsgestures_dataloader(dataset_path, test_dataset_path, batch_size):
    train_dl, test_dl = create_dataloader(dataset_path,
                                    batch_size=batch_size,
                                    size=[2 * 32 * 32],
                                    classes=[i for i in range(11)],
                                    sample_length_train=500000,
                                    sample_length_test=1500000, dt=10000,
                                    polarity=True, ds=4,
                                    shuffle_test=False, num_workers=0)

    _, test_dl_indistrib = create_dataloader(test_dataset_path,
                                             batch_size=batch_size,
                                             size=[2 * 32 * 32],
                                             classes=[i for i in range(11)],
                                             sample_length_train=500000,
                                             sample_length_test=1500000,
                                             dt=10000,
                                             polarity=True, ds=4,
                                             shuffle_train=False,
                                             shuffle_test=False,
                                             num_workers=0)

    return train_dl, test_dl, test_dl_indistrib


def make_twomoons_dataloader(batch_size, results_path):
    x_bin_train, y_bin_train, x_train, y_train \
        = make_moon_dataset(200, 100, 0.1, 10)
    train_dataset = CustomDataset(x_bin_train, y_bin_train)
    train_dl = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
    train_dl_noshuffle = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False)
    np.save(os.path.join(results_path, 'x_train'), x_train.numpy())
    np.save(os.path.join(results_path, 'y_train'), y_train.numpy())

    x_bin_test, x_test, y_test = make_moon_test(100, 100, 10)
    test_dataset = CustomDataset(x_bin_test, y_test)
    test_dl = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
    np.save(os.path.join(results_path, 'x_test'), x_test.numpy())
    np.save(os.path.join(results_path, 'y_test'), y_test.numpy())

    return train_dl, train_dl_noshuffle, test_dl

