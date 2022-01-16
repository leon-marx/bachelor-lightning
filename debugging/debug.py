import torch


if __name__ == "__main__":
    a = torch.arange(4 * 3 * 224 * 224).view(4, 3, 224, 224)
    b = torch.arange(4 * 3 * 224 * 224).view(4, 3, 224, 224) * 100
    c = torch.stack((a, b), dim=1).view(-1, 3, 224, 224)
    print(c.shape)
    print("done")