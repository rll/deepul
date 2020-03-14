from PIL import Image as PILImage
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy.ndimage
import cv2
import torch
from torchvision.utils import save_image
from deepul.utils import *

def load_mnist(num=None):
    # Read MNIST data
    data = input_data.read_data_sets("mnist", one_hot=True).train.images
    data = data.reshape(-1, 28, 28, 1).astype(np.float32)

    if num is None:
        return data
    else:
        return data[:num]

def get_colored_mnist(data):


    # Read Lena image
    lena = PILImage.open('lena.jpg')

    # Resize
    batch_resized = np.asarray([scipy.ndimage.zoom(image, (2.3, 2.3, 1), order=1) for image in data])
    
    # Extend to RGB
    batch_rgb = np.concatenate([batch_resized, batch_resized, batch_resized], axis=3)
    
    # Make binary
    batch_binary = (batch_rgb > 0.5)
    
    batch = np.zeros((data.shape[0], 28, 28, 3))
    
    for i in range(data.shape[0]):
        # Take a random crop of the Lena image (background)
        x_c = np.random.randint(0, lena.size[0] - 64)
        y_c = np.random.randint(0, lena.size[1] - 64)
        image = lena.crop((x_c, y_c, x_c + 64, y_c + 64))
        image = np.asarray(image) / 255.0

        # Invert the colors at the location of the number
        image[batch_binary[i]] = 1 - image[batch_binary[i]]
        
        batch[i] = cv2.resize(image, (0, 0), fx=28/64, fy=28/64, interpolation=cv2.INTER_AREA)
    return batch.transpose(0, 3, 1, 2)

def visualize_cyclegan_datasets():
    # mnist
    mnist = load_mnist(100)

    # colored_mnist
    colored_mnist = get_colored_mnist(mnist)
    print('here')
    show_samples(255 * mnist.transpose(0, 2, 3, 1), nrow=10, title='MNIST')
    # show_samples(colored_mnist, nrow=10, title='MNIST')

if __name__ == '__main__':
    # mnist_data = load_mnist()
    # colored_data = get_colored_mnist(mnist_data)
    visualize_cyclegan_datasets()
