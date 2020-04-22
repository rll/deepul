from deepul.hw4_helper import *

##################
##### Demo 3 #####
##################

def visualize_demo3_dataset():
    visualize_q1_dataset()

def demo3_save_results(part, fn):
    data = q1_data()
    losses, samples1, xs1, ys1, samples_end, xs_end, ys_end = fn(data)

    # loss plot
    plot_gan_training(losses, 'Demo 3{} Losses'.format(part), 'results/demo3{}_losses.png'.format(part))

    # samples
    q1_gan_plot(data, samples1, xs1, ys1, 'Demo 3{} Epoch 1'.format(part), 'results/demo3{}_epoch1.png'.format(part))
    q1_gan_plot(data, samples_end, xs_end, ys_end, 'Demo 3{} Final'.format(part), 'results/demo3{}_final.png'.format(part))

##################
##### Demo 4 #####
##################

def visualize_demo4_data():
    visualize_q2_data()

def demo4_save_results(fn):
    train_data = load_q2_data()
    train_data = train_data.data.transpose((0, 3, 1, 2)) / 255.0
    train_losses, samples = fn(train_data)
    plot_gan_training(train_losses, 'WGAN-GP Losses', 'results/demo4_losses.png')
    show_samples(samples[:100] * 255.0, fname='results/demo4_samples.png', title=f'CIFAR-10 generated samples')

##################
##### Demo 7 #####
##################

def visualize_demo7_data():
    visualize_q3_data()

def demo7_save_results(fn):
    train_data, test_data = load_q3_data()
    gan_losses, samples, reconstructions, pretrained_losses, random_losses = fn(train_data, test_data)

    plot_gan_training(gan_losses, 'Demo 7 Losses', 'results/demo7_gan_losses.png')
    plot_q3_supervised(pretrained_losses, random_losses, 'Linear classification losses', 'results/demo7_supervised_losses.png')
    show_samples(samples * 255.0, fname='results/demo7_samples.png', title='BiGAN generated samples')
    show_samples(reconstructions * 255.0, nrow=20, fname='results/demo7_reconstructions.png', title=f'BiGAN reconstructions')
    print('BiGAN final linear classification loss:', pretrained_losses[-1])
    print('Random encoder linear classification loss:', random_losses[-1])