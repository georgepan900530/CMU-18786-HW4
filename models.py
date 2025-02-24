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

import torch
import torch.nn as nn
from utils import SpectralNormWrapper


def up_conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=1,
    scale_factor=2,
    norm="batch",
    spectral_norm=False,
    activ=None,
):
    """Create a transposed-convolutional layer, with optional normalization."""
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode="nearest"))
    conv = nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding, bias=norm is None
    )

    if spectral_norm:
        layers.append(SpectralNormWrapper(conv))
    else:
        layers.append(conv)

    if norm == "batch":
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == "instance":
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == "relu":
        layers.append(nn.ReLU())
    elif activ == "leaky":
        layers.append(nn.LeakyReLU())
    elif activ == "tanh":
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


def conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=2,
    padding=1,
    norm="batch",
    init_zero_weights=False,
    spectral_norm=False,
    activ=None,
):
    """Create a convolutional layer, with optional normalization."""
    layers = []
    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=norm is None,
    )
    if init_zero_weights:
        conv_layer.weight.data = 0.001 * torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )

    if spectral_norm:
        layers.append(SpectralNormWrapper(conv_layer))
    else:
        layers.append(conv_layer)

    if norm == "batch":
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == "instance":
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == "relu":
        layers.append(nn.ReLU())
    elif activ == "leaky":
        layers.append(nn.LeakyReLU())
    elif activ == "tanh":
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):

    def __init__(self, noise_size, conv_dim=64):
        super().__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # According to the spec, it is better to directlyuse convolutional layer without any upsampling as the first layer and generate 4x4 output
        # Note that the input noise is in the shape of BSxnoise_sizex1x1
        self.up_conv1 = nn.ConvTranspose2d(
            noise_size, 4 * conv_dim, kernel_size=4, stride=1, padding=0
        )  # 100x1x1 -> 256x4x4
        self.up_conv2 = up_conv(
            4 * conv_dim, 2 * conv_dim, 3, 1, 1, norm="instance", activ="relu"
        )  # 256x4x4 -> 128x8x8
        self.up_conv3 = up_conv(
            2 * conv_dim, conv_dim, 3, 1, 1, norm="instance", activ="relu"
        )  # 128x8x8 -> 64x16x16
        self.up_conv4 = up_conv(
            conv_dim, 32, 3, 1, 1, norm="instance", activ="relu"
        )  # 64x16x16 -> 32x32x32
        self.up_conv5 = up_conv(32, 3, 3, 1, 1, activ="tanh")  # 32x32x32 -> 3x64x64

    def forward(self, z):
        """
        Generate an image given a sample of random noise.

        Input
        -----
            z: BS x noise_size x 1 x 1   -->  16x100x1x1

        Output
        ------
            out: BS x channels x image_width x image_height  -->  16x3x64x64
        """
        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################
        z = self.up_conv1(z)
        z = self.up_conv2(z)
        z = self.up_conv3(z)
        z = self.up_conv4(z)
        z = self.up_conv5(z)
        return z


class ResnetBlock(nn.Module):

    def __init__(self, conv_dim, norm, activ):
        super().__init__()
        self.conv_layer = conv(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=norm,
            activ=activ,
        )

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class DCDiscriminator(nn.Module):
    """Architecture of the discriminator network."""

    def __init__(self, conv_dim=64, norm="instance"):
        super().__init__()
        self.conv1 = conv(3, 32, 4, 2, 1, norm, False, "relu")  # 3x64x64 -> 32x32x32
        self.conv2 = conv(
            32, conv_dim, 4, 2, 1, norm, False, "relu"
        )  # 32x32x32 -> 64x16x16
        self.conv3 = conv(
            conv_dim, 2 * conv_dim, 4, 2, 1, norm, False, "relu"
        )  # 64x16x16 -> 128x8x8
        self.conv4 = conv(
            2 * conv_dim, 4 * conv_dim, 4, 2, 1, norm, False, "relu"
        )  # 128x8x8 -> 256x4x4
        self.conv5 = conv(
            4 * conv_dim, 1, 4, 1, 0, None, False, None
        )  # 256x4x4 -> 1x1x1

    def forward(self, x):
        """Forward pass, x is (B, C, H, W)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()


class DCGeneratorWithSpectralNorm(nn.Module):

    def __init__(self, noise_size, conv_dim=64):
        super().__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # According to the spec, it is better to directlyuse convolutional layer without any upsampling as the first layer and generate 4x4 output
        # Note that the input noise is in the shape of BSxnoise_sizex1x1
        self.up_conv1 = nn.ConvTranspose2d(
            noise_size, 4 * conv_dim, kernel_size=4, stride=1, padding=0
        )  # 100x1x1 -> 256x4x4
        self.up_conv2 = up_conv(
            4 * conv_dim, 2 * conv_dim, 3, 1, 1, norm="instance", activ="relu"
        )  # 256x4x4 -> 128x8x8
        self.up_conv3 = up_conv(
            2 * conv_dim, conv_dim, 3, 1, 1, norm="instance", activ="relu"
        )  # 128x8x8 -> 64x16x16
        self.up_conv4 = up_conv(
            conv_dim, 32, 3, 1, 1, norm="instance", activ="relu"
        )  # 64x16x16 -> 32x32x32
        self.up_conv5 = up_conv(32, 3, 3, 1, 1, activ="tanh")  # 32x32x32 -> 3x64x64

    def forward(self, z):
        """
        Generate an image given a sample of random noise.

        Input
        -----
            z: BS x noise_size x 1 x 1   -->  16x100x1x1

        Output
        ------
            out: BS x channels x image_width x image_height  -->  16x3x64x64
        """
        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################
        z = self.up_conv1(z)
        z = self.up_conv2(z)
        z = self.up_conv3(z)
        z = self.up_conv4(z)
        z = self.up_conv5(z)
        return z


class DCDiscriminatorWithSpectralNorm(nn.Module):
    """Architecture of the discriminator network."""

    def __init__(self, conv_dim=64, norm="instance"):
        super().__init__()
        self.conv1 = conv(
            3, 32, 4, 2, 1, norm=None, spectral_norm=True, activ="relu"
        )  # 3x64x64 -> 32x32x32
        self.conv2 = conv(
            32, conv_dim, 4, 2, 1, norm=None, spectral_norm=True, activ="relu"
        )  # 32x32x32 -> 64x16x16
        self.conv3 = conv(
            conv_dim, 2 * conv_dim, 4, 2, 1, norm=None, spectral_norm=True, activ="relu"
        )  # 64x16x16 -> 128x8x8
        self.conv4 = conv(
            2 * conv_dim,
            4 * conv_dim,
            4,
            2,
            1,
            norm=None,
            spectral_norm=True,
            activ="relu",
        )  # 128x8x8 -> 256x4x4
        self.conv5 = conv(
            4 * conv_dim, 1, 4, 1, 0, norm=None, spectral_norm=True
        )  # 256x4x4 -> 1x1x1

    def forward(self, x):
        """Forward pass, x is (B, C, H, W)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()
