# Fast Detection of Phase Transitions with Multi-Task Learning-by-Confusion

This repository contains a PyTorch implementation for the method introduced in our paper.

## Abstract of the article

Machine learning has been successfully used to study phase transitions. One
of the most popular approaches to identifying critical points from data without
prior knowledge of the underlying phases is the learning-by-confusion scheme.
As input, it requires system samples drawn from a grid of the parameter whose
change is associated with potential phase transitions. Up to now, the scheme
required training a distinct binary classifier for each possible splitting of the grid
into two sides, resulting in a computational cost that scales linearly with the
number of grid points. In this work, we propose and showcase an alternative
implementation that only requires the training of a single multi-class classifier.
Ideally, such multi-task learning eliminates the scaling with respect to the number
of grid points. In applications to the Ising model and an image dataset generated
with Stable Diffusion, we find significant speedups that, apart from small deviations,
correspond to this ideal case.

## How to use

* `generate_sd_images.py` is a short script that generates the Stable Diffusion dataset analyzed in our article (it requires [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1)).

* `script.ipynb` shows an example of how multi-task learning-by-confusion can be applied to an image dataset.

* `lbc_utils.py` implements helper functions in PyTorch that are convenient when using this method.

* `ising_net_architecture.py` contains the convolutional neural network used for analyzing the Ising dataset.
