import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

data = MNIST(root="data", download=False, train=True, transform=ToTensor())

plt.gray()

def make_plot():
    for i in range(4):
        ind = torch.randint(0, len(data), (1,)).item()
        img = data[ind][0].view(28, 28)
        plt.subplot(2, 2, i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

make_plot()