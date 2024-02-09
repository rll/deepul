# credits: Misha Laskin https://github.com/MishaLaskin/vqvae/blob/master/models/vqvae.py

import numpy as np
import torch
import torch.nn as nn

from deepul.models.decoder import Decoder
from deepul.models.encoder import Encoder
from deepul.models.quantizer import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(
        self,
        h_dim,
        res_h_dim,
        n_res_layers,
        n_embeddings,
        embedding_dim,
        beta,
        save_img_embedding_map=False,
    ):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1
        )
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Quantize an image x.

        Args:
            x (np.ndarray, dtype=int): Image to quantize. shape=(batch_size, 28, 28, 3). Values in [0, 3].

        Returns:
            np.ndarray: Quantized image. shape=(batch_size, 7, 7, 3)
        """
        x = torch.FloatTensor(x).permute(0, 3, 1, 2) - 1.5
        x = x.to(next(self.parameters()).device)
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        z_index = self.vector_quantization(z_e)
        return z_index

    def decode(self, z_index: np.ndarray) -> np.ndarray:
        """Decode a quantized image.

        Args:
            z_index (np.ndarray, dtype=int): Quantized image. shape=(batch_size, 7, 7). Values in [0, n_embeddings].

        Returns:
            np.ndarray: Decoded image. shape=(batch_size, 28, 28, 3). Values in [0, 3].
        """
        z_index = torch.LongTensor(z_index)
        z_index = z_index.to(next(self.parameters()).device)
        z_q = self.vector_quantization.embedding(z_index).permute(0, 3, 1, 2)
        x_hat = self.decoder(z_q) + 1.5
        return x_hat.permute(0, 2, 3, 1).detach().cpu().numpy()

    def forward(self, x, verbose=False):
        raise NotImplementedError
