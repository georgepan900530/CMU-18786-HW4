# CMU-18786-HW4

This is my homework 4 for CMU 18786 Introduction to Deep Learning Spring 2025.

## Part 1 - Deep Convolutional GAN

In this section, we are asked to built a vanilla GAN model according to the architecture given in the homework spec. In addition, we are asked to compare the output samples of GAN with basic/advanced data augmentation. Below is the command to run either basic or advanced version of DCGAN.

**Basic**

```
python vanilla_gan.py --[arguments]
```

**Advanced**

```
python vanilla_gan.py --data_preprocess "advanced" --[arguments]
```

## Part 2 - Architectures and Objective Functions

### Spectral Normalization

To run the spectral normalization variant of the DCGAN, one can simply follow the following command:

```
python vanilla_gan.py --model_type spectral --data_preprocess "advanced" --[arguments]
```

### Wasserstein GAN (GAN)

The implementaion of WGAN is referenced from [here](https://github.com/Zeleni9/pytorch-wgan). To train the WGAN, one can follow the following command:

```
python vanilla_gan.py --model_type WGAN --data_preprocess "advanced" --[arguments]
```

In addition to the original arguments, one can also specify `n_critic` or `gp_lambda` to adjust the training procedure.

### Least-Squares GAN (LSGAN)

The implementation of LSGAN is referenced from [here](https://sh-tsang.medium.com/review-lsgan-least-squares-generative-adversarial-networks-gan-bec12167e915). Run the following command to train LSGAN.

```
python vanilla_gan.py --model_type LSGAN --data_preprocess "advanced" --[arguments]
```
