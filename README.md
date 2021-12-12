# Efficient convolutional dictionary learning using preconditioned ADMM

This repository contains the implementation code for paper:

Efficient convolutional dictionary learning using preconditioned ADMM

## Abstract

Given training data, convolutional dictionary learning (CDL) seeks a translation-invariant sparse representation, which is characterized by a set of convolutional kernels. However, even a small training set with moderate sample size can render the optimization process both computationally challenging and memory starving. Under a biconvex optimization strategy for CDL, we propose to diagonally precondition the system matrices in the filter learning sub-problem that can be solved by the alternating direction method of multipliers (ADMM). This method leads to the substitution of matrix inversion (𝒪(n3)) and matrix multiplication (𝒪(n3)) involved in ADMM with an element-wise operation (𝒪(n)), which significantly reduces the computational complexity as well as the memory requirement.

## Requirements

- matlab2016a

## Run

run `learn_kernels_2D.m`

## Citation

```
@article{doi:10.1142/S0218001421510095,
author = {Zhang, Xuesong and Li, Baoping and Jiang, Jing},
title = {Efficient Convolutional Dictionary Learning Using Preconditioned ADMM},
journal = {International Journal of Pattern Recognition and Artificial Intelligence},
volume = {35},
number = {09},
pages = {2151009},
year = {2021},
doi = {10.1142/S0218001421510095},
URL = {https://doi.org/10.1142/S0218001421510095},
eprint = {https://doi.org/10.1142/S0218001421510095},
}
```

## Contact

If you have any questions, feel free to contact us through email (baoping_li@qq.com) or Github issues.

This repository will be constantly updated.

## Acknowledgements

This code is partly based on the open-source implementations of the paper: *F. Heide, W. Heidrich, G. Wetzstein "Fast and Flexible Convolutional Sparse Coding", IEEE Conference on Computer Vision and Pattern Recognition (CVPR Oral)*

