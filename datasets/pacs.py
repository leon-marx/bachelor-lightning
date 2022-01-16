import os
from PIL import Image
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
        self.data = []
        for domain in os.listdir(f"{self.data_dir}"):
            if domain in self.domains:
                for content in os.listdir(f"{self.data_dir}/{domain}"):
                    if content in self.contents:
                        for file in os.listdir(f"{self.data_dir}/{domain}/{content}"):
                            self.data.append(f"{domain}/{content}/{file}")
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = f"{self.data_dir}/{self.data[idx]}"
        # image = read_image(img_path)
        with Image.open(img_path) as image:
            image = self.transform(image)
            domain_name, content_name, filename = self.data[idx].split("/")
            domain = torch.nn.functional.one_hot(self.domain_dict[domain_name], num_classes=len(self.domains)).view(-1)
            content = torch.nn.functional.one_hot(self.content_dict[content_name], num_classes=len(self.contents)).view(-1)
            return image, domain, content, f"{domain_name}/{content_name}/{filename}"

        
class PACSDataModule(pl.LightningDataModule):
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
        self.domains = domains
        self.contents = contents
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_all = shuffle_all

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.pacs_train = PACSDataset(root=self.root, mode="train", domains=self.domains, contents=self.contents)
            self.pacs_val = PACSDataset(root=self.root, mode="val", domains=self.domains, contents=self.contents)
        if stage in (None, "test"):
            self.pacs_test = PACSDataset(root=self.root, mode="test", domains=self.domains, contents=self.contents)

    def train_dataloader(self):
        return DataLoader(self.pacs_train, batch_size=self.batch_size, shuffle=self.shuffle_all, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.pacs_val, batch_size=self.batch_size, shuffle=self.shuffle_all, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.pacs_test, batch_size=self.batch_size, shuffle=self.shuffle_all, num_workers=self.num_workers)
    

if __name__ == "__main__":
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    contents =  ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
    batch_size = 4
    dm = PACSDataModule(domains=domains, contents=contents, batch_size=batch_size)
    dm.setup()
    for (img, domain, content, fname) in dm.train_dataloader():
        print(img)
        print(domain)
        print(content)
        print(fname)