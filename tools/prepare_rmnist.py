from argparse import ArgumentParser
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import PIL
import matplotlib.pyplot as plt


def rotate_dataset(images, labels, angle):
    rotation = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Lambda(lambda x: transforms.functional.rotate(x, float(angle), fill=(0,),
        #     interpolation=PIL.Image.BILINEAR)),
        transforms.Lambda(lambda x: transforms.functional.rotate(x, float(angle), fill=(0,))),
        transforms.ToTensor()])

    x = torch.zeros(len(images), 1, 28, 28)
    for i in range(len(images)):
        x[i] = rotation(images[i])

    y = labels.view(-1)

    return x, y
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ratios", type=str, default="811")

    args = parser.parse_args()

    train_ratio = float(args.ratios[0]) * 0.1
    val_ratio = float(args.ratios[1]) * 0.1
    test_ratio = float(args.ratios[2]) * 0.1

    print("Configuration:")
    print(f"    train_ratio: {train_ratio}")
    print(f"    val_ratio: {val_ratio}")
    print(f"    test_ratio: {test_ratio}")
    print("")

    original_dataset_tr = MNIST("data/", train=True, download=True)
    original_dataset_te = MNIST("data/", train=False, download=True)
    original_images = torch.cat((original_dataset_tr.data,
                                    original_dataset_te.data))
    original_labels = torch.cat((original_dataset_tr.targets,
                                    original_dataset_te.targets))

    domains = [0, 15, 30, 45, 60, 75]
    contents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for j in contents:
        content_images = original_images[original_labels == j]

        # SHUFFLING
        N = content_images.shape[0]
        inds = np.arange(N)
        np.random.shuffle(inds)
        train_inds = inds[
            0
            :
            int(train_ratio * N)
        ]
        val_inds = inds[
            int(train_ratio * N)
            :
            int(train_ratio * N) + int(val_ratio * N)
        ]
        test_inds = inds[
            int(train_ratio * N) + int(val_ratio * N)
            :
            N
        ]

        print(f"    Size of train set: {len(train_inds)}")
        print(f"    Size of val set:   {len(val_inds)}")
        print(f"    Size of test set:  {len(test_inds)}")

        # Prepare train data
        for i, domain in enumerate(domains):
            my_indices = train_inds[i::len(domains)]
            labels = torch.ones((921,)) * j
            images = content_images[my_indices]
            print(images.shape, labels.shape)
            data = rotate_dataset(images, labels, domains[i])[0]
            print(data.min(), data.max())
            os.makedirs(f"data/RMNIST_train/{domain}/{j}")
            torch.save(data, f"data/RMNIST_train/{domain}/{j}/data.pt")
        # Prepare val data
        for i, domain in enumerate(domains):
            my_indices = val_inds[i::len(domains)]
            labels = torch.ones((921,)) * j
            images = content_images[my_indices]
            print(images.shape, labels.shape)
            data = rotate_dataset(images, labels, domains[i])[0]
            os.makedirs(f"data/RMNIST_val/{domain}/{j}")
            torch.save(data, f"data/RMNIST_val/{domain}/{j}/data.pt")
        # Prepare test data
        for i, domain in enumerate(domains):
            my_indices = test_inds[i::len(domains)]
            labels = torch.ones((921,)) * j
            images = content_images[my_indices]
            print(images.shape, labels.shape)
            data = rotate_dataset(images, labels, domains[i])[0]
            os.makedirs(f"data/RMNIST_test/{domain}/{j}")
            torch.save(data, f"data/RMNIST_test/{domain}/{j}/data.pt")

