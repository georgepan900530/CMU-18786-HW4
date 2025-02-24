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


# The implementation of spectral normalization is from the following link: https://gist.github.com/rosinality/a96c559d84ef2b138e486acf27b5a56e
class SpectralNorm:
    """
    Spectral Normalization

    In GAN training, spectral normalization is used to stabilize the training process. The spectral norm of a matrix W
    is defined as the largest singular value of W. In terms of GAN, spectral normalization is used to ensure
    that the Lipschitz constant of a neural network layer is bounded. The Lipschitz constant measures how much the output
    of a function can change relative to its input.

    """

    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        if weight_mat.is_cuda:
            u = u.cuda()
        v = weight_mat.t() @ u
        v = v / v.norm()
        u = weight_mat @ v
        u = u / u.norm()
        weight_sn = weight_mat / (u.t() @ weight_mat @ v)
        weight_sn = weight_sn.view(*size)

        return weight_sn, Variable(u.data)

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + "_orig", nn.Parameter(weight.data))
        input_size = weight.size(0)
        u = Variable(torch.randn(input_size, 1) * 0.1, requires_grad=False)
        setattr(module, name + "_u", u)
        setattr(module, name, fn.compute_weight(module)[0])

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + "_u", u)


def spectral_norm(module, name="weight"):
    SpectralNorm.apply(module, name)

    return module
