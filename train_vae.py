from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from torchvision.utils import save_image
from deepul.hw3_utils.lpips import LPIPS
from deepul.hw3_helper import *
import deepul.pytorch_util as ptu

ptu.set_gpu_mode(True)


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


def extract_patches_d(inputs: torch.Tensor, patch_size: int) -> torch.Tensor:
    inputs = inputs.permute(0, 2, 3, 1)
    B, H, W, C = inputs.shape

    assert H % patch_size == 0
    assert W % patch_size == 0
    P_H = H // patch_size
    P_W = W // patch_size
    x = inputs.reshape(B, P_H, patch_size, P_W, patch_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(-1, patch_size, patch_size, C).permute(0, 3, 1, 2)
    return x

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()

class Downsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1, bias=True):
        super(Downsample_Conv2d, self).__init__()
        conv = nn.Conv2d(in_dim, out_dim, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.conv = spectral_norm(conv)
        self.space_to_depth = SpaceToDepth(2)

    def forward(self, x):
        _x = self.space_to_depth(x)
        _x = sum(_x.chunk(4, dim=1)) / 4.0
        _x = self.conv(_x)
        return _x


class ResnetBlockDown(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), stride=1, n_filters=256):
        super(ResnetBlockDown, self).__init__()
        self.layers = nn.ModuleList([
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(in_dim, n_filters, kernel_size, stride=stride, padding=1)),
            nn.LeakyReLU(),
            Downsample_Conv2d(n_filters, n_filters, kernel_size),
            Downsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)
        ])

    def forward(self, x):
        _x = x
        for i in range(len(self.layers) - 1):
            _x = self.layers[i](_x)
        return self.layers[-1](x) + _x


class ResBlock(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super(ResBlock, self).__init__()
        self.layers = nn.ModuleList([
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(in_dim, n_filters, kernel_size, padding=1)),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(n_filters, n_filters, kernel_size, padding=1))
        ])

    def forward(self, x):
        _x = x
        for op in self.layers:
            _x = op(_x)
        return x + _x


from collections import OrderedDict
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        return x + self.net(x)

class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()

class Quantize(nn.Module):

    def __init__(self, size, code_dim):
        super().__init__()
        self.embedding = nn.Embedding(size, code_dim)
        self.embedding.weight.data.uniform_(-1./size,1./size)

        self.code_dim = code_dim
        self.size = size

    def forward(self, z):
        b, c, h, w = z.shape
        weight = self.embedding.weight

        flat_inputs = z.permute(0, 2, 3, 1).contiguous().view(-1, self.code_dim)
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * torch.mm(flat_inputs, weight.t()) \
                    + (weight.t() ** 2).sum(dim=0, keepdim=True)
        encoding_indices = torch.max(-distances, dim=1)[1]

        encoding_indices = encoding_indices.view(b, h, w)
        quantized = self.embedding(encoding_indices).permute(0, 3, 1, 2).contiguous()

        return quantized, (quantized - z).detach() + z, encoding_indices


class VAE(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
        )
        self.pre_quant = nn.Conv2d(256, 2 * latent_dim, 1)
        self.post_quant = nn.Conv2d(latent_dim, 256, 1)

        self.decoder = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x, return_dist=False):
        h = self.encoder(x)
        moments = self.pre_quant(h)
        posterior = DiagonalGaussianDistribution(moments)
        if return_dist:
            return posterior
        return posterior.sample()

    def decode(self, z):
        return self.decoder(self.post_quant(z))

    def forward(self, x):
        posterior = self.encode(x, return_dist=True)
        z = posterior.sample()
        x_tilde = self.decode(z)

        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        return x_tilde, kl_loss

    def loss(self, x):
        x = 2 * x - 1
        x_tilde, kl_loss = self(x)
        recon_loss = F.mse_loss(x_tilde, x)
        loss = recon_loss + 1e-6 * kl_loss
        return OrderedDict(loss=loss, recon_loss=recon_loss, reg_loss=kl_loss, x_gen=x_tilde)


class Discriminator(nn.Module):
    def __init__(self, n_filters, patchify=True):
        super(Discriminator, self).__init__()
        self.patchify = patchify
        network = [
            ResnetBlockDown(3, n_filters=n_filters),
            ResnetBlockDown(n_filters, n_filters=n_filters),
            ResBlock(n_filters, n_filters=n_filters),
            ResBlock(n_filters, n_filters=n_filters),
            nn.LeakyReLU()
        ]
        self.net = nn.Sequential(*network)
        self.fc = nn.Linear(n_filters, 1)
        self.sig = nn.Sigmoid()

    def forward(self, z):
        if self.patchify:
            z = extract_patches_d(z, 8)
        z = self.net(z)
        z = torch.sum(z, dim=(2, 3))
        return self.sig(self.fc(z))

class Solver(object):
    def __init__(self, train_data, test_data, n_epochs=1, batch_size=128, latent_dim=50, use_vit=False):
        self.log_interval = 100
        self.batch_size = batch_size
        self.use_vit = use_vit
        self.train_loader, self.test_loader = self.create_loaders(train_data, test_data)
        self.n_batches_in_epoch = len(self.train_loader)
        self.n_epochs = n_epochs
        self.curr_itr = 0
        self.latent_dim = latent_dim
        

    def build(self):
        self.d = Discriminator(128, patchify=not self.use_vit).to(ptu.device)
        self.d = Discriminator(128, patchify=True).to(ptu.device)
        self.g = VAE(latent_dim=2).to(ptu.device)
        self.g_optimizer = torch.optim.Adam(self.g.parameters(), lr=2e-4, betas=(0.5, 0.9))
        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer,
                                                             lambda epoch: (self.n_epochs - epoch) / self.n_epochs,
                                                             last_epoch=-1)
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=2e-4, betas=(0.5, 0.9))
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer,
                                                             lambda epoch: (self.n_epochs - epoch) / self.n_epochs,
                                                             last_epoch=-1)
        self.lpips_loss = LPIPS().to(ptu.device)
        

    def create_loaders(self, train_data, test_data):
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def get_discriminator_loss(self, x):
        x = 2 * x - 1
        x_fake = self.g(x)[0]
        x_real = x
        d_loss = - 0.5 * (self.d(x_real)).log().mean() - 0.5 * (1 - self.d(x_fake)).log().mean()
        return d_loss

    def get_discriminator_loss_g_only(self, x_fake):
        d_loss = -(self.d(x_fake)).log().mean()
        return d_loss

    def train_vqgan(self):
        d_loss_train = []
        g_l2_loss_train = []
        g_l2_loss_val = []
        g_lpips = []
        g_l2_loss_val.append(self.val())
        
        for epoch_i in tqdm(range(self.n_epochs), desc='Epoch'):
            epoch_i += 1

            self.d.train()
            self.g.train()
            
            self.batch_loss_history = []

            for batch_i, x in enumerate(tqdm(self.train_loader, desc='Batch', leave=False)):
                self.curr_itr += 1
                x = x.to(ptu.device).float()

                # # do a minibatch update
                self.d_optimizer.zero_grad()
                d_loss = self.get_discriminator_loss(x)
                d_loss.backward()
                self.d_optimizer.step()
                d_loss_train.append(d_loss.item())

                # generator and encoder update
                self.g_optimizer.zero_grad()
                vqgan_loss = self.g.loss(x)

                lpips = self.lpips_loss(vqgan_loss["x_gen"], 2 * x - 1).mean()
                g_loss_gan = self.get_discriminator_loss_g_only(vqgan_loss["x_gen"])
                
                g_loss = vqgan_loss["loss"] + g_loss_gan * 0.1  + 0.1 * lpips
                # if self.use_vit:
                #     g_loss += torch.nn.functional.l1_loss(vqgan_loss["x_gen"], 2 * x - 1).mean() * 0.1
                g_loss.backward()
                self.g_optimizer.step()

                g_l2_loss_train.append(vqgan_loss["recon_loss"].item())
                g_lpips.append(lpips.item())

                print(vqgan_loss['loss'].item(), vqgan_loss['recon_loss'].item(), vqgan_loss['reg_loss'].item(), d_loss.item())

                self.batch_loss_history.append(d_loss.item())

            # step the learning rate
            self.g_scheduler.step()
            self.d_scheduler.step()
            
            g_l2_loss_val.append(self.val())
            self.d.train()
            self.g.train()
            

        self.save_models('vqgan_weights.pt')
        return d_loss_train, g_lpips, g_l2_loss_train, g_l2_loss_val
        
    def val(self):
        self.g.eval()

        g_l2_loss_train = []
        for batch_i, x in enumerate(tqdm(self.test_loader, desc='Val', leave=False)):
            x = x.to(ptu.device).float()
            vqgan_loss = self.g.loss(x)
            g_l2_loss_train.append(vqgan_loss["recon_loss"].item())
                
        return np.mean(g_l2_loss_train)

    def save_models(self, filename):
        torch.save(self.g.state_dict(), "g_" + filename)
        torch.save(self.d.state_dict(), "d_" + filename)

    def load_models(self, filename):
        self.g.load_state_dict(torch.load("g_" + filename))
        self.d.load_state_dict(torch.load("d_" + filename))

def q3a(train_data, val_data, reconstruct_data):
    """
    train_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]
    val_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]
    reconstruct_data: An (100, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]. To be used for reconstruction

    Returns
    - a (# of training iterations,) numpy array of the discriminator train losses evaluated every minibatch
    - None or a (# of training iterations,) numpy array of the perceptual train losses evaluated every minibatch
    - a (# of training iterations,) numpy array of the l2 reconstruction evaluated every minibatch
    - a (# of epochs + 1,) numpy array of l2 reconstruction loss evaluated once at initialization and after each epoch on the val_data
    - a (100, 32, 32, 3) numpy array of reconstructions from your model in [0, 1] on the reconstruct_data.  
    """

    """ YOUR CODE HERE """
    
    solver = Solver(train_data, val_data, n_epochs=20)
    solver.build()
    discriminator_losses, l_pips_losses, l2_recon_train, l2_recon_test = solver.train_vqgan()

    solver.g.eval()
    solver.d.eval()
    with torch.no_grad():
        x_val = torch.tensor(val_data[0:100]).float().cuda()
        x_val = 2 *  x_val - 1
        reconstructions = (solver.g(x_val)[0].permute(0, 2, 3, 1).clamp(-1, 1).cpu().numpy() + 1) * 0.5
    torch.save(solver.g.state_dict(), 'vae_cifar10.pt')

    return discriminator_losses, l_pips_losses, l2_recon_train, l2_recon_test, reconstructions

q3_save_results(q3a, "a")
