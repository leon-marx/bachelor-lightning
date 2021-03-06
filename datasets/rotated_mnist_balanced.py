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

    
def collate_function(batch):
    images = []
    domains = []
    contents = []
    for i in range(batch[0][1][0].shape[0]):
        for minibatch in batch:
            images.append(minibatch[0][i])
            domains.append(minibatch[1][i])
            contents.append(minibatch[2][i])
    images = torch.stack(images)
    domains = torch.stack(domains)
    contents = torch.stack(contents)
    return images, domains, contents


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
        self.image_data = {domain: [] for domain in self.domains}
        self.domain_data = {domain: [] for domain in self.domains}
        self.content_data = {domain: [] for domain in self.domains}
        for domain in os.listdir(f"{self.data_dir}"):
            domain = int(domain)
            image_data_ = ()
            domain_data_ = ()
            content_data_ = ()
            if domain in self.domains:
                for content in os.listdir(f"{self.data_dir}/{domain}"):
                    content = int(content)
                    if content in self.contents:
                        imgs = torch.load(f"{self.data_dir}/{domain}/{content}/data.pt")
                        image_data_ += (imgs,)
                        domain_data_ += (torch.nn.functional.one_hot(self.domain_dict[domain], num_classes=len(self.domains)).view(1, -1),) * imgs.shape[0]
                        content_data_ += (torch.nn.functional.one_hot(self.content_dict[content], num_classes=len(self.contents)).view(1, -1),) * imgs.shape[0]
                image_data_ = torch.cat(image_data_, dim=0)
                domain_data_ = torch.cat(domain_data_, dim=0)
                content_data_ = torch.cat(content_data_, dim=0)
                torch.manual_seed(17 + domain)
                shuffle_inds = torch.randperm(len(image_data_))
                image_data_ = image_data_[shuffle_inds]
                domain_data_ = domain_data_[shuffle_inds]
                content_data_ = content_data_[shuffle_inds]
                self.image_data[domain] = image_data_
                self.domain_data[domain] = domain_data_
                self.content_data[domain] = content_data_
        self.transform = self.get_transform()

    def get_transform(self):
        transform = transforms.Compose([
            SetToTanhRange(),
        ])
        return transform

    def __len__(self):
        return max([len(self.image_data[domain]) for domain in self.domains])

    def __getitem__(self, idx):    
        images = []
        domains = []
        contents = []
        for d in self.domains:
            if idx >= len(self.image_data[d]):
                idx = torch.randint(low=0, high=len(self.image_data[d]), size=(1,)).item()
            image = self.transform(self.image_data[d][idx])
            domain = self.domain_data[d][idx]
            content = self.content_data[d][idx]
            images.append(image)
            domains.append(domain)
            contents.append(content)
        return images, domains, contents

        
class BalancedRMNISTDataModule(pl.LightningDataModule):
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
        return DataLoader(self.rmnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_function)
    
    def val_dataloader(self):
        return DataLoader(self.rmnist_val, batch_size=self.batch_size, shuffle=self.shuffle_all, num_workers=self.num_workers, collate_fn=collate_function)
    
    def test_dataloader(self):
        return DataLoader(self.rmnist_test, batch_size=self.batch_size, shuffle=self.shuffle_all, num_workers=self.num_workers, collate_fn=collate_function)
    

if __name__ == "__main__":
    domains = [0, 15, 30, 45, 60, 75]
    contents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batch_size = 4
    num_workers = 0
    root = "data"
    dm = BalancedRMNISTDataModule(root=root, domains=domains, contents=contents, batch_size=batch_size, num_workers=num_workers)
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
        for i in range(batch_size * len(domains)):
            plt.subplot(len(domains), batch_size, i+1)
            plt.imshow(img[i].view(28, 28))
            # plt.title([char.item() for char in domain[i]])
            # plt.ylabel([char.item() for char in content[i]])
            plt.xticks([])
            plt.yticks([])
        plt.show()
