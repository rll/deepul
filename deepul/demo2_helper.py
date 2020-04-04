import pickle
from os.path import join, exists
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from scipy.stats import norm
from sklearn.datasets import make_moons
import deepul.pytorch_util as ptu
from torchvision.utils import make_grid
from deepul.utils import get_data_dir, load_pickled_data, save_training_plot, show_samples
from deepul.hw2_helper import visualize_q3_data, get_q3_data

#######################################
######### Data for Flow Demos #########
#######################################

# def generate_1d_data(n, d):
#     rand = np.random.RandomState(0)
#     a = 0.3 + 0.1 * rand.randn(n)
#     b = 0.8 + 0.05 * rand.randn(n)
#     mask = rand.rand(n) < 0.5
#     samples = np.clip(a * mask + b * (1 - mask), 0.0, 1.0)
#     return np.digitize(samples, np.linspace(0.0, 1.0, d)).astype('float32')


# def generate_2d_data(n, dist):
#     import itertools
#     d1, d2 = dist.shape
#     pairs = list(itertools.product(range(d1), range(d2)))
#     idxs = np.random.choice(len(pairs), size=n, replace=True, p=dist.reshape(-1))
#     samples = [pairs[i] for i in idxs]
#
#     return np.array(samples).astype('float32')

def generate_1d_flow_data(n):
    assert n % 2 == 0
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n//2,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(n//2,))
    return np.concatenate([gaussian1, gaussian2])


def load_flow_demo_1(n_train, n_test, loader_args, visualize=True, train_only=False):
    # 1d distribution, mixture of two gaussians
    train_data, test_data = generate_1d_flow_data(n_train), generate_1d_flow_data(n_test)

    if visualize:
        plt.figure()
        x = np.linspace(-3, 3, num=100)
        densities = 0.5 * norm.pdf(x, loc=-1, scale=0.25) + 0.5 * norm.pdf(x, loc=0.5, scale=0.5)
        plt.figure()
        plt.plot(x, densities)
        plt.show()
        plt.figure()
        plt.hist(train_data, bins=50)
        # plot_hist(train_data, bins=50, title='Train Set')
        plt.show()

    train_dset, test_dset = NumpyDataset(train_data), NumpyDataset(test_data)
    train_loader, test_loader = data.DataLoader(train_dset, **loader_args), data.DataLoader(test_dset, **loader_args)

    if train_only:
        return train_loader
    return train_loader, test_loader

def load_flow_demo_2(n_train, n_test, loader_args, visualize=True, train_only=False, distribution='uniform'):
    if distribution == 'uniform':
        train_data = np.random.uniform(-2, 2, (n_train,))
        test_data = np.random.uniform(-2, 2, (n_test,))
        xs = np.linspace(-2, 2, 250)
        ys = np.ones_like(xs) * 0.25
    elif distribution == 'triangular':
        train_data = np.random.triangular(-2, -1.5, 2, (n_train,))
        test_data = np.random.triangular(-2, -1.5, 2, (n_test,))
        xs = np.linspace(-2, 2, 250)
        ys = np.zeros_like(xs)
        ys[xs < -1.5] = 2.0 + xs[xs < -1.5]
        ys[xs >= -1.5] = (2 - xs[xs >= -1.5]) / 7
        plt.plot(xs, ys)
    elif distribution == 'complex':
        xs = np.linspace(0, 1, 100)
        ys = np.tanh(np.sin(8 * np.pi * xs) + 3 * np.power(xs, 0.5))
        train_data = np.random.choice(np.linspace(-2, 2, 100), size=n_train, p=ys / ys.sum())
        test_data = np.random.choice(np.linspace(-2, 2, 100), size=n_test, p=ys / ys.sum())
        xs = np.linspace(-2, 2, 100)
        ys = ys / ys.sum() / 0.04
    else:
        raise NotImplementedError

    if visualize:
        plt.figure()
        plt.plot(xs, ys)
        plt.hist(train_data, bins=50, density=True)
        plt.title(distribution)
        plt.show()

    train_dset, test_dset = NumpyDataset(train_data), NumpyDataset(test_data)
    train_loader, test_loader = data.DataLoader(train_dset, **loader_args), data.DataLoader(test_dset, **loader_args)

    if train_only:
        return train_loader
    return train_loader, test_loader


# Demo 3: Autoregressive Flows in 2D

def load_smiley_face(n):
    count = n
    rand = np.random.RandomState(0)
    a = [[-1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    b = [[1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    c = np.c_[2 * np.cos(np.linspace(0, np.pi, count // 3)),
              -np.sin(np.linspace(0, np.pi, count // 3))]
    c += rand.randn(*c.shape) * 0.2
    data_x = np.concatenate([a, b, c], axis=0)
    data_y = np.array([0] * len(a) + [1] * len(b) + [2] * len(c))
    perm = rand.permutation(len(data_x))
    return data_x[perm], data_y[perm]

def load_half_moons(n):
    return make_moons(n_samples=n, noise=0.1)

def make_scatterplot(points, title=None, filename=None):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=1)
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig("q1_{}.png".format(filename))


def load_flow_demo_3(n_train, n_test, loader_args, visualize=True, train_only=False, distribution='face'):
    if distribution == 'face':
        train_data, train_labels = load_smiley_face(n_train)
        test_data, test_labels = load_smiley_face(n_test)
    elif distribution == 'moons':
        train_data, train_labels = load_half_moons(n_train)
        test_data, test_labels = load_half_moons(n_test)
    else:
        raise NotImplementedError

    if visualize:
        plt.figure()
        plt.scatter(train_data[:, 0], train_data[:, 1], s=1, c=train_labels)
        plt.title(distribution)
        plt.show()

    train_dset, test_dset = NumpyDataset(train_data), NumpyDataset(test_data)
    train_loader, test_loader = data.DataLoader(train_dset, **loader_args), data.DataLoader(test_dset, **loader_args)

    if train_only:
        return train_loader
    return train_loader, test_loader, train_labels, test_labels


# Demo 4: Parameter Sharing (Autoregressive Flows)

def visualize_demo4_data():
    data_dir = get_data_dir(2)
    train_data, test_data = load_pickled_data(join(data_dir, 'shapes.pkl'))
    name = 'Shape'

    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs] * 255
    show_samples(images, title=f'{name} Samples')

def demo4_save_results(fn):
    data_dir = get_data_dir(2)
    train_data, test_data = load_pickled_data(join(data_dir, 'shapes.pkl'))

    train_losses, test_losses, samples, floored_samples = fn(train_data, test_data)
    samples = samples.astype('float')
    floored_samples = floored_samples.astype('float')

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    save_training_plot(train_losses, test_losses, f'Q2 Dataset Train Plot', 'results/demo4_train_plot.png')
    show_samples(samples * 255.0)
    show_samples(floored_samples * 255.0, title='Samples with Flooring')


def demo6_save_results(fn, part):
    data_dir = get_data_dir(2)
    train_data, test_data = load_pickled_data(join(data_dir, 'celeb.pkl'))

    train_losses, test_losses, samples, interpolations = fn(train_data, test_data)
    samples = samples.astype('float')
    interpolations = interpolations.astype('float')

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    save_training_plot(train_losses, test_losses, f'RealNVP Train Plot',
                       f'results/demo6_{part}_train_plot.png')
    show_samples(samples * 255.0, f'results/demo6_{part}_samples.png')
    show_samples(interpolations * 255.0, f'results/demo6_{part}_interpolations.png', nrow=6, title='Interpolations')


class NumpyDataset(data.Dataset):

    def __init__(self, array, transform=None):
        super().__init__()
        self.array = array
        self.transform = transform

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        x = self.array[index]
        if self.transform:
            x = self.transform(x)
        return x

######################################################
######### Visualization Utils for Flow Demos #########
######################################################

def plot_hist(data, bins=10, xlabel='x', ylabel='Probability', title='', density=None):
    bins = np.concatenate((np.arange(bins) - 0.5, [bins - 1 + 0.5]))

    plt.figure()
    plt.hist(data, bins=bins, density=True)

    if density:
        plt.plot(density[0], density[1], label='distribution')
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_2d_dist(dist, title='Learned Distribution'):
    plt.figure()
    plt.imshow(dist)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x0')
    plt.show()

def plot_train_curves(epochs, train_losses, test_losses, title=''):
    x = np.linspace(0, epochs, len(train_losses))
    plt.figure()
    plt.plot(x, train_losses, label='train_loss')
    if test_losses:
        plt.plot(x, test_losses, label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

def visualize_batch(batch_tensor, nrow=8, title='', figsize=None):
    grid_img = make_grid(batch_tensor, nrow=nrow)
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()


def plot_1d_continuous_dist(density, xlabel='x', ylabel="Density", title=''):
    plt.figure()
    plt.plot(density[0], density[1], label='distribution')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def visualize_demo1_flow(train_loader, initial_flow, final_flow):
    plt.figure(figsize=(10,5))
    train_data = ptu.FloatTensor(train_loader.dataset.array)

    # before:
    plt.subplot(231)
    plt.hist(ptu.get_numpy(train_data), bins=50)
    plt.title('True Distribution of x')

    plt.subplot(232)
    x = ptu.FloatTensor(np.linspace(-3, 3, 200))
    z, _ = initial_flow.flow(x)
    plt.plot(ptu.get_numpy(x), ptu.get_numpy(z))
    plt.title('Flow x -> z')

    plt.subplot(233)
    z_data, _ = initial_flow.flow(train_data)
    plt.hist(ptu.get_numpy(z_data), bins=50)
    plt.title('Empirical Distribution of z')

    # after:
    plt.subplot(234)
    plt.hist(ptu.get_numpy(train_data), bins=50)
    plt.title('True Distribution of x')

    plt.subplot(235)
    x = ptu.FloatTensor(np.linspace(-3, 3, 200))
    z, _ = final_flow.flow(x)
    plt.plot(ptu.get_numpy(x), ptu.get_numpy(z))
    plt.title('Flow x -> z')

    plt.subplot(236)
    z_data, _ = final_flow.flow(train_data)
    plt.hist(ptu.get_numpy(z_data), bins=50)
    plt.title('Empirical Distribution of z')

    plt.tight_layout()

def plot_demo2_losses(losses):
    # taken with modification from matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    n_layers = np.flip(np.arange(1, 1 + len(losses)), axis=0)
    n_components = np.arange(1, 1 + losses.shape[1])

    fig, ax = plt.subplots()
    flipped_losses = np.flip(losses, axis=0)

    im = ax.imshow(flipped_losses)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(n_components)))
    ax.set_yticks(np.arange(len(n_layers)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(n_components)
    ax.set_yticklabels(n_layers)

    # Loop over data dimensions and create text annotations.
    for i in range(len(n_layers)):
        for j in range(len(n_components)):
            text = ax.text(j, i, "{:0.2f}".format(flipped_losses[i, j]), ha="center", va="center", color="w")
    ax.set_xlabel("Number of components per layer")
    ax.set_ylabel("Number of layers")
    ax.set_title("Nats/dim using varying composition schemes")
    fig.tight_layout()

def visualize_demo6_data():
    visualize_q3_data()

def get_demo6_data():
    return get_q3_data()