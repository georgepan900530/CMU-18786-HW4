# CMU CMU 18-780/6 Homework 4
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator        --> Used in the vanilla GAN in Part 1
#   - DCDiscriminator    --> Used in both the vanilla GAN in Part 1
# For the assignment, you are asked to create the architectures of these
# three networks by filling in the __init__ and forward methods in the
# DCGenerator, DCDiscriminator classes.
# Feel free to add and try your own models

import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.detach().numpy()


def create_dir(directory):
    """Creates a directory if it does not already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


class SpectralNorm:
    """
    Spectral Normalization

    In GAN training, spectral normalization is used to stabilize the training process. The spectral norm of a matrix W
    is defined as the largest singular value of W. In terms of GAN, spectral normalization is used to ensure
    that the Lipschitz constant of a neural network layer is bounded. The Lipschitz constant measures how much the output
    of a function can change relative to its input.

    Parameters
    -----
    module: nn.Module
        The module to apply spectral normalization to.
    name: str
        The name of the weight parameter to apply spectral normalization to.
    n_power_iterations: int
        The number of power iterations to use to compute the spectral norm.
    eps: float
        The epsilon value to use to prevent division by zero.
    """

    def __init__(self, module, name="weight", n_power_iterations=1, eps=1e-12):
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps

        # Get the weight parameters
        try:
            w = getattr(self.module, self.name)
        except AttributeError:
            raise ValueError(f"{name} is not an attribute of the module {module}")

        # Initialize u and v which are the left and right singular vectors
        self.register_buffer("u", torch.randn(w.shape[0]))
        self.register_buffer("v", None)

    def _update_u_v(self):
        weight = getattr(self.module, self.name).data
        u = self.u
        v = self.v

        for _ in range(self.n_power_iterations):
            # Compute v = (W^T)u / ||(W^T)u||
            v = F.normalize(torch.mv(weight.t(), u), p=2, dim=0)
            # Compute u = (Wv) / ||Wv||
            u = F.normalize(torch.mv(weight, v), p=2, dim=0)

        self.u.data = u
        self.v.data = v

    def _apply_spectral_norm(self):
        weight = getattr(self.module, self.name)
        u = self.u
        v = self.v

        # Compute the spectral norm
        sigma = torch.dot(u, torch.mv(weight, v))
        # Normalize the weight
        weight.data = weight.data / sigma

    def forward(self, *args, **kwargs):
        self._update_u_v()
        self._apply_spectral_norm()
        return self.module(*args, **kwargs)


class SpectralNormWrapper(nn.Module):
    """
    Wrapper for SpectralNorm
    """

    def __init__(self, module, name="weight", n_power_iterations=1, eps=1e-12):
        super(SpectralNormWrapper, self).__init__()
        self.spectral_norm = SpectralNorm(module, name, n_power_iterations, eps)

    def forward(self, *args, **kwargs):
        return self.spectral_norm.forward(*args, **kwargs)
