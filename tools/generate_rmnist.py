from argparse import ArgumentParser
import os
import torch

from datasets.rotated_mnist import RMNISTDataModule
from models.cvae_v3 import CVAE_v3

if __name__ == "__main__":
    # Parser
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--domain_transfer", action="store_true", default=False)
    parser.add_argument("--content_transfer", action="store_true", default=False)
    parser.add_argument("--generate", action="store_true", default=False)
    parser.add_argument("--domains", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    domains = sorted([int(char) for char in args.domains])
    contents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    dm = RMNISTDataModule(root="data", domains=domains, contents=contents,
                        batch_size=args.batch_size, num_workers=20)
    dm.setup()
    if args.model == "CVAE_v3":
        model = CVAE_v3.load_from_checkpoint(args.ckpt_path)

    domain_dict = {domain: torch.LongTensor([i]) for i, domain in enumerate(domains)}
    content_dict = {content: torch.LongTensor([i]) for i, content in enumerate(contents)}

    with torch.no_grad():
        model.eval()
        if args.domain_transfer and not args.content_transfer:
            for domain in domains:
                data = {content: [] for content in contents}
                for batch in dm.train_dataloader():
                    dec_domains = torch.cat((torch.nn.functional.one_hot(domain_dict[domain], num_classes=len(domains)),) * batch[1].shape[0], dim=0).to(model.device)
                    dec_contents = batch[2]
                    transfers = model.transfer(batch[0], batch[1], batch[2], dec_domains, dec_contents)
                    for j in range(args.batch_size):
                        data[int(torch.argmax(dec_contents[j]).item())].append(transfers[j])

                for content in contents:
                    data[content] = torch.cat(data[content], dim=0)
                    print(data[content].shape)
                    torch.save(data[content], f"data/variants/RMNIST_augmented/RMNIST_train/{domain}/{content}/data.pt")
        if not args.domain_transfer and args.content_transfer:
            for content in contents:
                data = {domain: [] for domain in domains}
                for batch in dm.train_dataloader():
                    dec_domains = torch.cat((torch.nn.functional.one_hot(domain_dict[domain], num_classes=len(domains)),) * batch[1].shape[0], dim=0).to(model.device)
                    dec_contents = torch.cat((torch.nn.functional.one_hot(content_dict[content], num_classes=len(contents)),) * batch[1].shape[0], dim=0).to(model.device)
                    transfers = model.transfer(batch[0], batch[1], batch[2], dec_domains, dec_contents)
                    data.append(transfers)

                data = torch.cat(data, dim=0)
                print(data.shape)
                torch.save(data, f"data/variants/RMNIST_augmented/RMNIST_train/{domain}/{content}/data.pt")
        if args.domain_transfer and args.content_transfer:
            for domain in domains:
                for content in contents:
                    data = []
                    for batch in dm.train_dataloader():
                        dec_domains = torch.cat((torch.nn.functional.one_hot(domain_dict[domain], num_classes=len(domains)),) * batch[1].shape[0], dim=0).to(model.device)
                        dec_contents = torch.cat((torch.nn.functional.one_hot(content_dict[content], num_classes=len(contents)),) * batch[1].shape[0], dim=0).to(model.device)
                        transfers = model.transfer(batch[0], batch[1], batch[2], dec_domains, dec_contents)
                        data.append(transfers)

                    data = torch.cat(data, dim=0)
                    print(data.shape)
                    torch.save(data, f"data/variants/RMNIST_augmented/RMNIST_train/{domain}/{content}/data.pt")

        model.train()



