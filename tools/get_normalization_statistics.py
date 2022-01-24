import torch
from tqdm import tqdm
from datasets import pacs
from datasets.pacs import PACSDataModule
from argparse import ArgumentParser



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--domain", type=str)
    args = parser.parse_args()
    # domains = ["art_painting", "cartoon", "photo", "sketch"]
    gather_data = True
    batch_size = 1
    N = 50

    if gather_data:
        dm = PACSDataModule(
            root="data", 
            domains= [args.domain],
            contents=["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"],
            batch_size=batch_size,
            num_workers=20,
            normalize=False)
        dm.setup()

        mean = 0.0
        std = 0.0
        for i in tqdm(range(N)):
            channels_sum, channels_squared_sum, num_batches = 0.0, 0.0, 0.0
            for batch in tqdm(dm.train_dataloader(), leave=False):
                # Mean over batch, height and width, but not over the channels
                channels_sum += torch.mean(batch[0], dim=[0,2,3])
                channels_squared_sum += torch.mean(batch[0]**2, dim=[0,2,3])
                num_batches += 1
        
            mean += channels_sum / num_batches / N
            # std = sqrt(E[X^2] - (E[X])^2)
            std += (channels_squared_sum / num_batches - mean ** 2) ** 0.5 / N

        torch.save(mean, f"logs/{args.domain}_mean_statistics.pt")
        torch.save(std, f"logs/{args.domain}_std_statistics.pt")
        print(args.domain)
        print("mean:", mean)
        print("std:", std)
        print("")