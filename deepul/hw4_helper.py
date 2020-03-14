import numpy as np
import torchvision
import torchvision.transforms as transforms

from .utils import *


def plot_gan_training(losses, title, fname):
    plt.figure()
    n_itr = len(losses)
    xs = np.arange(n_itr)

    plt.plot(xs, losses, label='loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    savefig(fname)

def q1_gan_plot(data, samples, xs, ys, title, fname):
    plt.figure()
    plt.hist(samples, bins=50, density=True, alpha=0.7, label='fake')
    plt.hist(data, bins=50, density=True, alpha=0.7, label='real')

    plt.plot(xs, ys, label='discrim')
    plt.legend()
    plt.title(title)
    savefig(fname)


######################
##### Question 1 #####
######################

def q1_data(n=20000):
    assert n % 2 == 0
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n//2,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(n//2,))
    data = (np.concatenate([gaussian1, gaussian2]) + 1).reshape([-1, 1])
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return 2 * scaled_data -1

def visualize_q1_dataset():
    data = q1_data()
    plt.hist(data, bins=50, alpha=0.7, label='train data')
    plt.legend()
    plt.show()


def q1_save_results(part, fn):
    data = q1_data()
    losses, samples1, xs1, ys1, samples_end, xs_end, ys_end = fn(data)

    # loss plot
    plot_gan_training(losses, 'Q1{} Losses'.format(part), 'q1{}_losses.png'.format(part))

    # samples
    q1_gan_plot(data, samples1, xs1, ys1, 'Q1{} Epoch 1'.format(part), 'q1{}_epoch1.png'.format(part))
    q1_gan_plot(data, samples_end, xs_end, ys_end, 'Q1{} Final'.format(part), 'q1{}_final.png'.format(part))

######################
##### Question 2 #####
######################

def calculate_is(samples):
    return 0

def load_q2_data():
    train_data = torchvision.datasets.CIFAR10("./data", transform=torchvision.transforms.ToTensor(),
                                              download=True, train=True)
    return train_data

def visualize_q2_data():
    train_data = load_q2_data()
    imgs = train_data.data[:100]
    show_samples(imgs, title=f'CIFAR-10 Samples')

def q2_save_results(fn):
    train_data = load_q2_data()

    train_losses, samples = fn(train_data)

    print("Inception score:", calculate_is(samples))
    plot_gan_training(train_losses, 'Q2 Losses', 'q2_losses.png')
    show_samples(samples[:100] * 255.0, fname='q2_samples.png', title=f'CIFAR-10 generated samples')

######################
##### Question 3 #####
######################

def load_q3_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_data, test_data

def visualize_q3_data():
    train_data, _ = load_q3_data()
    imgs = train_data.data[:100]
    show_samples(imgs.reshape([100, 28, 28, 1]) * 255.0, title=f'MNIST samples')

def plot_q3_supervised(pretrained_losses, random_losses, title, fname):
    plt.figure()
    xs = np.arange(len(pretrained_losses))
    plt.plot(xs, pretrained_losses, label='bigan')
    xs = np.arange(len(random_losses))
    plt.plot(xs, random_losses, label='random init')
    plt.legend()
    plt.title(title)
    savefig(fname)

def q3_save_results(fn):
    train_data, test_data = load_q3_data()
    gan_losses, samples, reconstructions, pretrained_losses, random_losses = fn(train_data, test_data)

    plot_gan_training(gan_losses, 'Q3 Losses', 'q3_gan_losses.png')
    plot_q3_supervised(pretrained_losses, random_losses, 'Linear classification losses', 'q3_supervised_losses.png')
    show_samples(samples * 255.0, fname='q3_samples.png', title='BiGAN generated samples')
    show_samples(reconstructions * 255.0, nrow=20, fname='q3_reconstructions.png', title=f'BiGAN reconstructions')
    print('BiGAN final linear classification loss:', pretrained_losses[-1])
    print('Random encoder linear classification loss:', random_losses[-1])


