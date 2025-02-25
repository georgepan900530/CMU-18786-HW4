# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# This is the main training file for the first part of the assignment.
#
# Usage:
# ======
#    To train with the default hyperparamters
#    (saves results to checkpoints_vanilla/ and samples_vanilla/):
#       python vanilla_gan.py

import argparse
import os

import imageio
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import utils
from data_loader import get_data_loader
from models import *


policy = "color,translation,cutout"

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_models(G, D):
    """Prints model information for the generators and discriminators."""
    print("                    G                  ")
    print("---------------------------------------")
    print(G)
    print("---------------------------------------")

    print("                    D                  ")
    print("---------------------------------------")
    print(D)
    print("---------------------------------------")


def create_model(opts):
    """Builds the generators and discriminators."""
    if opts.model_type == "vanilla" or opts.model_type == "WGAN":
        G = DCGenerator(noise_size=opts.noise_size, conv_dim=opts.conv_dim)
        D = DCDiscriminator(conv_dim=opts.conv_dim)
    elif opts.model_type == "spectral":
        G = DCGenerator(noise_size=opts.noise_size, conv_dim=opts.conv_dim)
        D = DCDiscriminatorWithSpectralNorm(conv_dim=opts.conv_dim)

    print_models(G, D)

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        print("Models moved to GPU.")

    return G, D


def create_image_grid(array, ncols=None):
    """Useful docstring (insert there)."""
    num_images, channels, cell_h, cell_w = array.shape

    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros((cell_h * nrows, cell_w * ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w, :] = (
                array[i * ncols + j].transpose(1, 2, 0)
            )

    if channels == 1:
        result = result.squeeze()
    return result


def checkpoint(iteration, G, D, opts):
    """Save the parameters of the generator G and discriminator D."""
    G_path = os.path.join(opts.checkpoint_dir, "G_iter%d.pkl" % iteration)
    D_path = os.path.join(opts.checkpoint_dir, "D_iter%d.pkl" % iteration)
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)


def save_samples(G, fixed_noise, iteration, opts):
    generated_images = G(fixed_noise)
    generated_images = utils.to_data(generated_images)

    grid = create_image_grid(generated_images)
    grid = np.uint8(255 * (grid + 1) / 2)

    # merged = merge_images(X, fake_Y, opts)
    path = os.path.join(opts.sample_dir, "sample-{:06d}.png".format(iteration))
    imageio.imwrite(path, grid)
    print("Saved {}".format(path))


def save_images(images, iteration, opts, name):
    grid = create_image_grid(utils.to_data(images))

    path = os.path.join(opts.sample_dir, "{:s}-{:06d}.png".format(name, iteration))
    grid = np.uint8(255 * (grid + 1) / 2)
    imageio.imwrite(path, grid)
    print("Saved {}".format(path))


def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Variable of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Variable of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return utils.to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)


def training_loop(train_dataloader, opts):
    """Runs the training loop.
    * Saves checkpoints every opts.checkpoint_every iterations
    * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    G, D = create_model(opts)

    # Create optimizers for the generators and discriminators
    if opts.optimizer == "Adam":
        g_optimizer = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
        d_optimizer = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])
    elif opts.optimizer == "RMSprop":
        g_optimizer = optim.RMSprop(G.parameters(), opts.lr)
        d_optimizer = optim.RMSprop(D.parameters(), opts.lr)

    # Generate fixed noise for sampling from the generator
    fixed_noise = sample_noise(opts.batch_size, opts.noise_size)  # B N 1 1

    iteration = 1

    total_train_iters = opts.num_epochs * len(train_dataloader)

    for _ in range(opts.num_epochs):

        for batch in train_dataloader:

            real_images = batch
            real_images = utils.to_var(real_images)

            # TRAIN THE DISCRIMINATOR
            # 1. Compute the discriminator loss on real images
            D_real = D(real_images)
            # Note that the output of the disciminator does not go through a sigmoid
            if opts.model_type != "WGAN":
                D_real_loss = F.binary_cross_entropy_with_logits(
                    D_real, torch.ones_like(D_real)
                )

            # 2. Sample noise
            noise = sample_noise(opts.batch_size, opts.noise_size)

            # 3. Generate fake images from the noise
            fake_images = G(noise)

            # 4. Compute the discriminator loss on the fake images
            D_fake = D(fake_images)
            if opts.model_type != "WGAN":
                D_fake_loss = F.binary_cross_entropy_with_logits(
                    D_fake, torch.zeros_like(D_fake)
                )
                D_total_loss = D_real_loss + D_fake_loss
            else:
                D_total_loss = torch.mean(D_fake) - torch.mean(D_real)

            # update the discriminator D
            d_optimizer.zero_grad()
            D_total_loss.backward()
            d_optimizer.step()

            # We need to clip the weights of the discriminator to enforce Lipschitz constraint
            if opts.model_type == "WGAN":
                for p in D.parameters():
                    p.data.clamp_(-opts.clip_value, opts.clip_value)

            # TRAIN THE GENERATOR
            # 1. Sample noise
            noise = sample_noise(opts.batch_size, opts.noise_size)

            # 2. Generate fake images from the noise
            fake_images = G(noise)

            # 3. Compute the generator loss
            if opts.model_type != "WGAN":
                G_loss = F.binary_cross_entropy_with_logits(
                    D(fake_images), torch.ones_like(D(fake_images))
                )
            else:
                G_loss = -torch.mean(D(fake_images))

            # update the generator G
            g_optimizer.zero_grad()
            G_loss.backward()
            g_optimizer.step()

            # Print the log info
            if iteration % opts.log_step == 0:
                print(
                    "Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | "
                    "D_fake_loss: {:6.4f} | G_loss: {:6.4f}".format(
                        iteration,
                        total_train_iters,
                        D_real_loss.item(),
                        D_fake_loss.item(),
                        G_loss.item(),
                    )
                )
                logger.add_scalar("D/fake", D_fake_loss, iteration)
                logger.add_scalar("D/real", D_real_loss, iteration)
                logger.add_scalar("D/total", D_total_loss, iteration)
                logger.add_scalar("G/total", G_loss, iteration)

            # Save the generated samples
            if iteration % opts.sample_every == 0:
                save_samples(G, fixed_noise, iteration, opts)
                save_images(real_images, iteration, opts, "real")

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            iteration += 1


def training_loop_wgan(train_dataloader, opts):
    """Runs the WGAN training loop.
    * Saves checkpoints every opts.checkpoint_every iterations
    * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    G, D = create_model(opts)

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])

    # Generate fixed noise for sampling from the generator
    fixed_noise = sample_noise(opts.batch_size, opts.noise_size)  # B N 1 1

    iteration = 1

    total_train_iters = opts.num_epochs * len(train_dataloader)

    for _ in range(opts.num_epochs):

        for batch in train_dataloader:

            real_images = batch
            real_images = utils.to_var(real_images)

            # TRAIN THE DISCRIMINATOR
            # for i in range(opts.n_critic):
            # 1. Compute the discriminator loss on real images
            D_real = D(real_images)
            # 2. Sample noise
            noise = sample_noise(real_images.shape[0], opts.noise_size)

            # 3. Generate fake images from the noise
            fake_images = G(noise)

            # 4. Compute the discriminator loss on the fake images
            D_fake = D(fake_images)
            D_loss = torch.mean(D_fake) - torch.mean(D_real)

            # Gradient Penalty - Better stability than clipping weights
            # Interpolate between real and fake images
            epsilon = torch.rand(
                real_images.shape[0], 1, 1, 1, device=real_images.device
            )
            interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images
            interpolated_images.requires_grad_(True)

            # Get critic output on interpolated images
            D_interpolated = D(interpolated_images)

            # Compute gradients of D_interpolated with respect to interpolated_images
            grad_outputs = torch.ones_like(D_interpolated)
            gradients = torch.autograd.grad(
                outputs=D_interpolated,
                inputs=interpolated_images,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            # Reshape gradients and compute their L2 norm for each sample in the batch
            gradients = gradients.view(gradients.size(0), -1)
            gradient_norm = gradients.norm(2, dim=1)

            # Compute the penalty as the squared difference from 1
            gradient_penalty = ((gradient_norm - 1) ** 2).mean()

            # Set the gradient penalty coefficient (lambda)
            lambda_gp = 10

            D_total_loss = D_loss + lambda_gp * gradient_penalty

            # update the discriminator D
            d_optimizer.zero_grad()
            D_total_loss.backward()
            d_optimizer.step()

            # According to the algorithm 1 in WGAN paper, we need to update the critic n_critic times befor updating the generator
            # This is equivalent to update the generator after n_critic iterations
            if iteration % opts.n_critic == 0:
                # TRAIN THE GENERATOR
                # 1. Sample noise
                noise = sample_noise(opts.batch_size, opts.noise_size)

                # 2. Generate fake images from the noise
                fake_images = G(noise)

                # 3. Compute the generator loss
                G_loss = -torch.mean(D(fake_images))

                # update the generator G
                g_optimizer.zero_grad()
                G_loss.backward()
                g_optimizer.step()

            # Print the log info
            if iteration % opts.log_step == 0:
                print(
                    "Iteration [{:4d}/{:4d}] | D_total_loss: {:6.4f} | "
                    "G_loss: {:6.4f}".format(
                        iteration,
                        total_train_iters,
                        D_total_loss.item(),
                        G_loss.item(),
                    )
                )
                logger.add_scalar("D/total", D_total_loss, iteration)
                logger.add_scalar("G/total", G_loss, iteration)

            # Save the generated samples
            if iteration % opts.sample_every == 0:
                save_samples(G, fixed_noise, iteration, opts)
                save_images(real_images, iteration, opts, "real")

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            iteration += 1


def main(opts):
    """Loads the data and starts the training loop."""

    # Create a dataloader for the training images
    dataloader = get_data_loader(opts.data, opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    if opts.model_type == "WGAN":
        training_loop_wgan(dataloader, opts)
    else:
        training_loop(dataloader, opts)


def create_parser():
    """Creates a parser for command-line arguments."""
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--conv_dim", type=int, default=32)
    parser.add_argument("--noise_size", type=int, default=100)
    parser.add_argument("--model_type", type=str, default="vanilla")
    parser.add_argument("--clip_value", type=float, default=0.01)

    # Training hyper-parameters
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--n_critic", type=int, default=1)

    # Data sources
    parser.add_argument("--data", type=str, default="cat/grumpifyBprocessed")
    parser.add_argument("--data_preprocess", type=str, default="basic")
    parser.add_argument("--ext", type=str, default="*.png")

    # Directories and checkpoint/sample iterations
    parser.add_argument("--checkpoint_dir", default="checkpoints_vanilla")
    parser.add_argument("--sample_dir", type=str, default="vanilla")
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=200)
    parser.add_argument("--checkpoint_every", type=int, default=400)

    return parser


if __name__ == "__main__":
    parser = create_parser()
    opts = parser.parse_args()

    batch_size = opts.batch_size
    opts.sample_dir = os.path.join(
        "output/",
        opts.sample_dir,
        "%s_%s" % (os.path.basename(opts.data), opts.data_preprocess),
    )

    if os.path.exists(opts.sample_dir):
        cmd = "rm %s/*" % opts.sample_dir
        os.system(cmd)
    logger = SummaryWriter(opts.sample_dir)
    print(opts)
    main(opts)
