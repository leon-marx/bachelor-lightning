import os
from PIL import Image
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SetToTanhRange(object):
    """
    Shift torch.Tensor from [0, 1] to [-1, 1] range.
    """
    def __call__(self, sample):
        return 2.0 * sample - 1.0


class RMNISTDataset(Dataset):
    def __init__(self, root, mode, domains, contents):
        """
        root: str, root folder where RMNIST is located
        mode: str, choose one: "train", "val" or "test"
        domains: list of int [0, 15, 30, 45, 60, 75]
        contents: list of int [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        """
        super().__init__()
        self.mode = mode
        self.domains = domains
        self.contents = contents
        self.domain_dict = {domain: torch.LongTensor([i]) for i, domain in enumerate(self.domains)}
        self.content_dict = {content: torch.LongTensor([i]) for i, content in enumerate(self.contents)}
        self.data_dir = f"{root}/RMNIST_{mode}"
        self.image_data = ()
        self.domain_data = ()
        self.content_data = ()
        for domain in os.listdir(f"{self.data_dir}"):
            if domain in self.domains:
                for content in os.listdir(f"{self.data_dir}/{domain}"):
                    content = int(content)
                    if content in self.contents:
                        imgs = torch.load(f"{self.data_dir}/{domain}/{content}/data.pt")
                        self.image_data += (imgs,)
                        self.domain_data += (torch.nn.functional.one_hot(self.domain_dict[domain], num_classes=len(self.domains)).view(1, -1),) * imgs.shape[0]
                        self.content_data += (torch.nn.functional.one_hot(self.content_dict[content], num_classes=len(self.contents)).view(1, -1),) * imgs.shape[0]
            self.image_data = torch.cat(self.image_data, dim=0)
            self.domain_data = torch.cat(self.domain_data, dim=0)
            self.content_data = torch.cat(self.content_data, dim=0)
            torch.manual_seed(17 + domain)
            shuffle_inds = torch.randperm(len(self.image_data))
            self.image_data = self.image_data[shuffle_inds]
            self.domain_data = self.domain_data[shuffle_inds]
            self.content_data = self.content_data[shuffle_inds]
        self.transform = self.get_transform()

    def get_transform(self):
        transform = transforms.Compose([
            SetToTanhRange(),
        ])
        return transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = self.transform(self.image_data[idx])
        domain = self.domain_data[idx]
        content = self.content_data[idx]
        return image, domain, content

        
class RMNISTDataModule(pl.LightningDataModule):
    def __init__(self, root, domains, contents, batch_size, num_workers, shuffle_all=False):
        """
        root: str, root folder where RMNIST is located
        domains: list of int [0, 15, 30, 45, 60, 75]
        contents: list of int [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        batch_size: int, batch_size to use for the dataloaders
        num_workers: int, how many workers to use for the dataloader
        shuffle_all: bool, if True val and test dataloaders are shuffled as well
        """
        super().__init__()
        self.root = root
        self.domains = domains
        self.contents = contents
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_all = shuffle_all

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.rmnist_train = RMNISTDataset(root=self.root, mode="train", domains=self.domains, contents=self.contents)
            self.rmnist_val = RMNISTDataset(root=self.root, mode="val", domains=self.domains, contents=self.contents)
        if stage in (None, "test"):
            self.rmnist_test = RMNISTDataset(root=self.root, mode="test", domains=self.domains, contents=self.contents)

    def train_dataloader(self):
        return DataLoader(self.rmnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.rmnist_val, batch_size=self.batch_size, shuffle=self.shuffle_all, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.rmnist_test, batch_size=self.batch_size, shuffle=self.shuffle_all, num_workers=self.num_workers)
    

if __name__ == "__main__":
    argument_domains = ["0", "1", "2", "3", "4"]
    domain_dict = {
            "a": [0, 15, 30, 45, 60, 75],
            "0": [0],
            "1": [15],
            "2": [30],
            "3": [45],
            "4": [60],
            "5": [75],
        }
    domains = []
    for key in argument_domains:
        domains += domain_dict[key]
    domains = sorted(domains)
    contents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    print(domains)
    print(argument_domains)
    batch_size = 4
    num_workers = 0
    root = "data"
    dm = RMNISTDataModule(root=root, domains=domains, contents=contents, batch_size=batch_size, num_workers=num_workers)
    dm.setup()
    import numpy as np
    import matplotlib.pyplot as plt
    for (img, domain, content) in dm.train_dataloader():
        print(img)
        print(domain)
        print(content)
        print(img.shape)
        print(domain.shape)
        print(content.shape)
        print(img.min(), img.max())
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(img[i].view(28, 28))
            plt.title([char.item() for char in domain[i]])
            plt.ylabel([char.item() for char in content[i]])
            plt.xticks([])
            plt.yticks([])
        plt.show()
