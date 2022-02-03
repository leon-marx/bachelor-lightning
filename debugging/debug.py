import matplotlib.pyplot as plt
import torch
import umap
import numpy as np

if __name__ == "__main__":
    out_channels = [256]
    num_domains = 3
    num_classes = 7
    beta = 1e-5

    y_mmd = torch.randn(size=(49*out_channels[0]))

    mmd = 0

    n = int(y_mmd.shape[0] / num_domains)
    labeled_y = [y_mmd[i*n:(i+1)*n] for i in range(num_domains)]
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    for i in range(len(labeled_y)):
        for j in range(i+1):
            k = 0
            for x1 in labeled_y[i]:
                for x2 in labeled_y[j]:
                    e = torch.exp(-(x1 - x2) ** 2).mean()
                    for sigma in sigmas:
                        k += e ** sigma * beta
            if i == j:
                mmd += k / n ** 2
            else:
                mmd -= 2 * k / n ** 2
    print(mmd)