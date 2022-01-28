import os
from PIL import Image
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torchvision import transforms as TT

from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder, RandomResizedCropRGBImageDecoder
from ffcv import transforms as FT


class SetToTanhRange(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 2.0 * x - 1.0


class Sort(torch.nn.Module):
    def __init__(self, num_domains):
        super().__init__()
        self.num_domains = num_domains

    def forward(self, x):
        s = []
        for i in range(self.num_domains):
            for minibatch in x:
                s.append(minibatch[i])
        s = torch.stack(s)
        return s


class PACSDataset(Dataset):
    def __init__(self, root, mode, domains, contents):
        """
        root: str, root folder where PACS is located
        mode: str, choose one: "train", "val" or "test"
        domains: list of str ["art_painting", "cartoon", "photo", "sketch"]
        contents: list of str ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        """
        super().__init__()
        self.mode = mode
        self.domains = domains
        self.contents = contents
        self.domain_dict = {domain: torch.LongTensor([i]) for i, domain in enumerate(self.domains)}
        self.content_dict = {content: torch.LongTensor([i]) for i, content in enumerate(self.contents)}
        self.data_dir = f"{root}/PACS_{mode}"
        self.data = {domain: [] for domain in self.domains}
        for domain in os.listdir(f"{self.data_dir}"):
            if domain in self.domains:
                for content in os.listdir(f"{self.data_dir}/{domain}"):
                    if content in self.contents:
                        for file in os.listdir(f"{self.data_dir}/{domain}/{content}"):
                            self.data[domain].append(f"{domain}/{content}/{file}")

    def __len__(self):
        return max([len(self.data[domain]) for domain in self.domains])

    def __getitem__(self, idx):
        images = []
        domains = []
        contents = []
        for domain in self.domains:
            if idx >= len(self.data[domain]):
                idx = torch.randint(low=0, high=len(self.data[domain]), size=(1,)).item()
            img_path = f"{self.data_dir}/{self.data[domain][idx]}"
            # image = read_image(img_path)
            with Image.open(img_path) as image:
                image = self.transform(image)
                domain_name, content_name, _ = self.data[domain][idx].split("/")
                domain = torch.nn.functional.one_hot(self.domain_dict[domain_name], num_classes=len(self.domains)).view(-1)
                content = torch.nn.functional.one_hot(self.content_dict[content_name], num_classes=len(self.contents)).view(-1)
                images.append(image)
                domains.append(domain)
                contents.append(content)
        print(torch.stack(images), torch.stack(domains), torch.stack(contents))
        return torch.stack(images), torch.stack(domains), torch.stack(contents)

        
class BalancedPACSDataModule(pl.LightningDataModule):
    def __init__(self, root, domains, contents, batch_size, num_workers, shuffle_all=False):
        """
        root: str, root folder where PACS is located
        domains: list of str ["art_painting", "cartoon", "photo", "sketch"]
        contents: list of str ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        batch_size: int, batch_size to use for the dataloaders
        num_workers: int, how many workers to use for the dataloader
        shuffle_all: bool, if True val and test dataloaders are shuffled as well
        """
        super().__init__()
        self.root = root
        self.domains = sorted(domains)
        self.contents = contents
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_all = shuffle_all
        domain_string = ""
        for d in self.domains:
            domain_string += d[0]
        self.train_path = f"{root}/fast_paths/PACS_{domain_string}_train"
        self.val_path = f"{root}/fast_paths/PACS_{domain_string}_val"
        self.test_path = f"{root}/fast_paths/PACS_{domain_string}_test"
        self.order_dict = {
            True: OrderOption.RANDOM,
            False: OrderOption.SEQUENTIAL
        }
        self.pipeline = {
            "images": [
                RandomResizedCropRGBImageDecoder(224, scale=(0.7, 1.0), ratio=(0.995, 1.005)),
                FT.RandomHorizontalFlip(),
                TT.ColorJitter(0.3, 0.3, 0.3, 0.3),
                TT.RandomGrayscale(),
                FT.ToTensor(),
                SetToTanhRange(),
                Sort(len(self.domains))
            ],
            "domains": [NDArrayDecoder(), Sort(len(self.domains)), FT.ToTensor()],
            "contents": [NDArrayDecoder(), Sort(len(self.domains)), FT.ToTensor()]
        }

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.pacs_train = PACSDataset(root=self.root, mode="train", domains=self.domains, contents=self.contents)
            self.pacs_val = PACSDataset(root=self.root, mode="val", domains=self.domains, contents=self.contents)
        if stage in (None, "test"):
            self.pacs_test = PACSDataset(root=self.root, mode="test", domains=self.domains, contents=self.contents)

    def train_dataloader(self):
        return Loader(self.train_path, batch_size=self.batch_size, num_workers=self.num_workers, order=self.order_dict[True], pipelines=self.pipeline)
    
    def val_dataloader(self):
        return Loader(self.val_path, batch_size=self.batch_size, num_workers=self.num_workers, order=self.order_dict[self.shuffle_all], pipelines=self.pipeline)
    
    def test_dataloader(self):
        return Loader(self.test_path, batch_size=self.batch_size, num_workers=self.num_workers, order=self.order_dict[self.shuffle_all], pipelines=self.pipeline)
    

if __name__ == "__main__":
    domains = ["art_painting", "cartoon", "photo"]
    contents =  ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
    batch_size = 4
    num_workers = 20
    root = "data"
    dm = BalancedPACSDataModule(root=root, domains=domains, contents=contents, batch_size=batch_size, num_workers=num_workers)
    dm.setup()
    # import numpy as np
    # import matplotlib.pyplot as plt
    # def gauss(x):
    #     return 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2)
    # xx = np.linspace(-4, 4, 100)
    for (img, domain, content) in dm.train_dataloader():
        print(img.shape)
        print(domain.shape)
        print(content.shape)
        print(domain)
        # print(fname)
        # plt.hist(img.flatten().numpy(), density=True)
        # plt.plot(xx, gauss(xx))
        # plt.show()
        # plt.close()
        print(img.min(), img.max())
    # torch.save(data, "debugging/data.pt")
