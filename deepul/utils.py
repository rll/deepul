import os
import pickle
import textwrap
from os.path import dirname, exists, join

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def savefig(fname: str, show_figure: bool = True) -> None:
    if not exists(dirname(fname)):
        os.makedirs(dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()


def save_training_plot(
    train_losses: np.ndarray, test_losses: np.ndarray, title: str, fname: str
) -> None:
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label="train loss")
    plt.plot(x_test, test_losses, label="test loss")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    savefig(fname)


def save_timing_plot(
    time_1: np.ndarray,
    time_2: np.ndarray,
    title: str,
    fname: str,
    time1_label: str,
    time2_label: str,
) -> None:
    plt.figure()

    plt.plot(time_1, label=time1_label)
    plt.plot(time_2, label=time2_label)
    plt.legend()
    plt.title(title)
    plt.xlabel("sample step")
    plt.ylabel("seconds")
    savefig(fname)


def save_scatter_2d(data: np.ndarray, title: str, fname: str) -> None:
    plt.figure()
    plt.title(title)
    plt.scatter(data[:, 0], data[:, 1])
    savefig(fname)


def save_distribution_1d(
    data: np.ndarray, distribution: np.ndarray, title: str, fname: str
):
    d = len(distribution)

    plt.figure()
    plt.hist(data, bins=np.arange(d) - 0.5, label="train data", density=True)

    x = np.linspace(-0.5, d - 0.5, 1000)
    y = distribution.repeat(1000 // d)
    plt.plot(x, y, label="learned distribution")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("Probability")
    plt.legend()
    savefig(fname)


def save_distribution_2d(true_dist: np.ndarray, learned_dist: np.ndarray, fname: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.imshow(true_dist)
    ax1.set_title("True Distribution")
    ax1.axis("off")
    ax2.imshow(learned_dist)
    ax2.set_title("Learned Distribution")
    ax2.axis("off")
    savefig(fname)


def show_samples(
    samples: np.ndarray, fname: str = None, nrow: int = 10, title: str = "Samples"
):
    import torch
    from torchvision.utils import make_grid

    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")

    if fname is not None:
        savefig(fname)
    else:
        plt.show()


def load_pickled_data(fname: str, include_labels: bool = False):
    with open(fname, "rb") as f:
        data = pickle.load(f)

    train_data, test_data = data["train"], data["test"]
    if "mnist.pkl" in fname or "shapes.pkl" in fname:
        # Binarize MNIST and shapes dataset
        train_data = (train_data > 127.5).astype("uint8")
        test_data = (test_data > 127.5).astype("uint8")
    if "celeb.pkl" in fname:
        train_data = train_data[:, :, :, [2, 1, 0]]
        test_data = test_data[:, :, :, [2, 1, 0]]
    if include_labels:
        return train_data, test_data, data["train_labels"], data["test_labels"]
    return train_data, test_data


def load_colored_mnist_text(file_path):
    """
    Load colored MNIST data from a pickle file.

    Parameters:
    file_path (str): Path to the pickle file containing the dataset.

    Returns:
    tuple: Tuple containing training and test datasets.
    """
    with open(file_path, "rb") as f:
        (
            colored_train_data,
            colored_test_data,
            colored_train_labels,
            colored_test_labels,
        ) = pickle.load(f)
    return (
        colored_train_data,
        colored_test_data,
        colored_train_labels,
        colored_test_labels,
    )


def get_data_dir(hw_number: int):
    return join('deepul', 'homeworks', f'hw{hw_number}', 'data')


def quantize(images: np.ndarray, n_bits: int = 8):
    images = np.floor(images / 256.0 * 2**n_bits)
    return images.astype("uint8")


def load_text_data(pickle_file_path: str):
    test_set_size = 10

    # Load the data from the pickle file
    with open(pickle_file_path, "rb") as file:
        data = pickle.load(file)

    # Split the data
    train_set = data[:-test_set_size]
    test_set = data[-test_set_size:]

    return train_set, test_set

def save_text_to_plot(text_samples, filename, image_width=600, image_height=900):
    scale_factor = 2

    # Adjusted dimensions and settings
    image_width *= scale_factor
    image_height *= scale_factor
    background_color = (255, 255, 255)
    title_font_size = 30 * scale_factor
    text_font_size = 16 * scale_factor
    padding = 20 * scale_factor
    spacing = 4 * scale_factor
    title_spacing = 40 * scale_factor
    title_font = ImageFont.truetype("arial.ttf", title_font_size)
    text_font = ImageFont.truetype("arial.ttf", text_font_size)

    # Create an image canvas
    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)
    title = "Text Samples"
    draw.text((padding, padding), title, font=title_font, fill="black")

    # Starting position for drawing text
    current_h = padding + title_font_size + title_spacing

    # Add each text sample to the image
    for i, text in enumerate(text_samples, start=1):
        # Label for each text sample
        label = f"Text Sample {i}"
        draw.text((padding, current_h), label, font=text_font, fill="black")
        current_h += text_font_size + spacing

        # Wrap and draw the text
        wrapped_text = textwrap.fill(text, width=26 * scale_factor)
        draw.text((padding, current_h), wrapped_text, font=text_font, fill="black")

        # Update the Y position for the next text
        current_h += (text_font_size + spacing) * len(wrapped_text.split("\n")) + title_spacing

    dpi = 300
    plt.figure(figsize=(image_width / dpi, image_height / dpi), dpi=dpi)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(filename, dpi=dpi)
    plt.show()
