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


def spectral_norm(weight, u=None, n_power_iterations=1, eps=1e-12):
    """
    Applies spectral normalization to a convolutional weight tensor.

    Args:
        weight (torch.Tensor): Weight tensor with shape (out_channels, in_channels, k_h, k_w).
        u (torch.Tensor, optional): The vector for power iteration. If None, it will be initialized.
        n_power_iterations (int): Number of power iterations.
        eps (float): Small epsilon for numerical stability.

    Returns:
        weight_sn (torch.Tensor): Spectrally normalized weight.
        u (torch.Tensor): The updated u vector.
    """
    # Reshape the weight to a 2D matrix: (out_channels, in_channels * k_h * k_w)
    out_channels = weight.size(0)
    weight_mat = weight.view(out_channels, -1)

    # Initialize u if not provided
    if u is None:
        u = torch.randn(out_channels, device=weight.device)
        u = F.normalize(u, dim=0, eps=eps)

    # Power iteration: update u and estimate largest singular value
    for _ in range(n_power_iterations):
        v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=eps)
        u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=eps)

    sigma = torch.dot(u, torch.matmul(weight_mat, v))
    weight_sn = weight / sigma
    return weight_sn, u


# Spectrally Normalized 2D Convolution Layer
class SpectralNormConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        n_power_iterations=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.n_power_iterations = n_power_iterations
        # Register buffer for u
        self.register_buffer("u", None)

    def compute_weight(self):
        # Get the original weight
        weight = self.conv.weight
        # Apply spectral normalization
        weight_sn, self.u = spectral_norm(weight, self.u, self.n_power_iterations)
        # Update the weight
        self.conv.weight.data = weight_sn

    def forward(self, x):
        # Compute the spectrally normalized weight
        self.compute_weight()
        # Perform the convolution with the normalized weight
        return self.conv(x)
