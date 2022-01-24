import torch
from tqdm import tqdm
from datasets import pacs
from datasets.pacs import PACSDataModule


if __name__ == "__main__":
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    gather_data = True
    batch_size = 1
    N = 50

    if gather_data:
        for domain in domains:
            test_dm = PACSDataModule(
                root="data", 
                domains= [domain],
                contents=["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"],
                batch_size=batch_size,
                num_workers=20,
                normalize=False)
            test_dm.setup()
            true_total = torch.zeros(size=(len(test_dm.train_dataloader()), 3, 224, 224))
            for i in range(N):
                dm = PACSDataModule(
                    root="data", 
                    domains= [domain],
                    contents=["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"],
                    batch_size=batch_size,
                    num_workers=20,
                    normalize=False)
                dm.setup()
                total = torch.zeros(size=(3, 224, 224))
                counter = 0.0
                for batch in tqdm(dm.train_dataloader()):
                    total += batch[0].sum(dim=0)
                    counter += batch_size
                total /= counter
                true_total += total / N
            torch.save(true_total, f"logs/{domain}_statistics.pt")
    for domain in domains:
        tot = torch.load(f"logs/{domain}_statistics.pt")
        print(domain)
        print("mean:", tot.mean(dim=[0, 2, 3]))
        print("std:", tot.std(dim=[0, 2, 3]))
        print("")