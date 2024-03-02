import numpy as np
import matplotlib.pyplot as plt


######################
##### Question 1 #####
######################


def q1_data(n=20000):
    centers = np.array([[5, 5], [-5, 5], [0, -5]])
    st_devs = np.array([[1.0, 1.0], [0.2, 0.2], [3.0, 0.5]])
    labels = np.random.randint(0, 3, size=(n,), dtype='int32')
    x = np.random.randn(n, 2) * st_devs[labels] + centers[labels]
    return x.astype('float32')


def visualize_q1_dataset():
    data = q1_data()
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


def q1_save_results(part, fn):
    data = q1_data()