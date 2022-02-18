from argparse import ArgumentParser
import os
import torch
from tqdm import tqdm

from datasets.rotated_mnist import RMNISTDataModule
from models.cvae_v3 import CVAE_v3

def get_activation(act_string):
    if "ReLU" in act_string:
        activation = torch.nn.ReLU()
    if "SELU" in act_string:
        activation = torch.nn.SELU()
    if "ELU" in act_string:
        activation = torch.nn.ELU()
    return activation

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

    argument_domains = args.domains.split(",")
    argument_ckpt_paths = args.ckpt_path.split(",")
    for i, arg_domain in enumerate(argument_domains):
        print(f"Starting transfer on {arg_domain}")
        arg_ckpt_path = argument_ckpt_paths[i]
        domain_string = arg_domain
        domains = sorted([int(char) for char in arg_domain])
        contents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        dm = RMNISTDataModule(root="data", domains=domains, contents=contents,
                            batch_size=args.batch_size, num_workers=20)
        dm.setup()

        out_channels = []
        lr = 1e-4
        hparam_path = ""
        for dir in arg_ckpt_path.split("/")[:-2]:
            hparam_path += dir + "/"
        hparam_path += "hparams.yaml"
        with open(hparam_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split(" ")
                print(line)
                if "data" in line[0]:
                    data = str(line[1])
                if "num_domains" in line[0]:
                    num_domains = int(line[1])
                if "num_contents" in line[0]:
                    num_contents = int(line[1])
                if "latent_size" in line[0]:
                    latent_size = int(line[1])
                if "depth" in line[0]:
                    depth = int(line[1])
                if "kernel_size" in line[0]:
                    kernel_size = int(line[1])
                if "activation" in line[0]:
                    activation = get_activation(str(line[1]))
                if "downsampling" in line[0]:
                    downsampling = str(line[1])
                if "upsampling" in line[0]:
                    upsampling = str(line[1])
                if "dropout" in line[0]:
                    dropout = line[1] == "true"
                if "batch_norm" in line[0]:
                    batch_norm = line[1] == "true"
                if "loss_mode" in line[0]:
                    loss_mode = str(line[1])
                if "lamb" in line[0]:
                    lamb = float(line[1])
                if "no_bn_last" in line[0]:
                    no_bn_last = line[1] == "true"
                if "-" in line[0]:
                    try:
                        val = int(line[1])
                        out_channels.append(val)
                    except ValueError:
                        print("ValueError:", line[1])

        print("Loading Model with:")
        print(f"  data: {data}")
        print(f"  num_domains: {num_domains}")
        print(f"  num_contents: {num_contents}")
        print(f"  latent_size: {latent_size}")
        print(f"  lr: {lr}")
        print(f"  depth: {depth}")
        print(f"  out_channels: {out_channels}")
        print(f"  kernel_size: {kernel_size}")
        print(f"  activation: {activation}")
        print(f"  downsampling: {downsampling}")
        print(f"  upsampling: {upsampling}")
        print(f"  dropout: {dropout}")
        print(f"  batch_norm: {batch_norm}")
        print(f"  loss_mode: {loss_mode}")
        print(f"  lamb: {lamb}")
        print(f"  no_bn_last: {no_bn_last}")

        if args.model == "CVAE_v3":
            model = CVAE_v3.load_from_checkpoint(
                arg_ckpt_path,
                data=data,
                num_domains=num_domains,
                num_contents=num_contents,
                latent_size=latent_size,
                lr=lr,
                depth=depth,
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                downsampling=downsampling,
                upsampling=upsampling,
                dropout=dropout,
                batch_norm=batch_norm,
                loss_mode=loss_mode,
                lamb=lamb,
                no_bn_last=no_bn_last)

        domain_dict = {domain: torch.LongTensor([i]) for i, domain in enumerate(domains)}
        content_dict = {content: torch.LongTensor([i]) for i, content in enumerate(contents)}

        with torch.no_grad():
            model.eval()
            if args.domain_transfer and not args.content_transfer:
                for domain in domains:
                    data = {content: [] for content in contents}
                    print("")
                    print(f"Transferring to {domain} degrees:")
                    for batch in tqdm(dm.train_dataloader()):
                        dec_domains = torch.cat((torch.nn.functional.one_hot(domain_dict[domain], num_classes=len(domains)),) * batch[1].shape[0], dim=0).to(model.device)
                        dec_contents = batch[2]
                        transfers = model.transfer(batch[0], batch[1], batch[2], dec_domains, dec_contents)
                        for j in range(args.batch_size):
                            data[int(torch.argmax(dec_contents[j]).item())].append(transfers[j])
                    for content in contents:
                        data_save_dir = f"data/variants/RMNIST_augmented/RMNIST_train_{domain_string}/{domain}/{content}/data.pt"
                        os.makedirs(data_save_dir[:-8], exist_ok=True)
                        data[content] = torch.cat(data[content], dim=0)
                        print(data[content].shape)
                        torch.save(data[content], data_save_dir)

            if not args.domain_transfer and args.content_transfer:
                for content in contents:
                    data = {domain: [] for domain in domains}
                    print("")
                    print(f"Transferring to number {content}:")
                    for batch in tqdm(dm.train_dataloader()):
                        dec_domains = batch[1]
                        dec_contents = torch.cat((torch.nn.functional.one_hot(content_dict[content], num_classes=len(contents)),) * batch[1].shape[0], dim=0).to(model.device)
                        transfers = model.transfer(batch[0], batch[1], batch[2], dec_domains, dec_contents)
                        for j in range(args.batch_size):
                            data[int(torch.argmax(dec_contents[j]).item())].append(transfers[j])
                    for domain in domains:
                        data_save_dir = f"data/variants/RMNIST_augmented/RMNIST_train_{domain_string}/{domain}/{content}/data.pt"
                        os.makedirs(data_save_dir[:-8], exist_ok=True)
                        data[domain] = torch.cat(data[domain], dim=0)
                        print(data[domain].shape)
                        torch.save(data[domain], data_save_dir)

            if args.domain_transfer and args.content_transfer:
                for domain in domains:
                    for content in contents:
                        data = []
                        print("")
                        print(f"Transferring to {domain} degrees and number {content}:")
                        for batch in tqdm(dm.train_dataloader()):
                            dec_domains = torch.cat((torch.nn.functional.one_hot(domain_dict[domain], num_classes=len(domains)),) * batch[1].shape[0], dim=0).to(model.device)
                            dec_contents = torch.cat((torch.nn.functional.one_hot(content_dict[content], num_classes=len(contents)),) * batch[1].shape[0], dim=0).to(model.device)
                            transfers = model.transfer(batch[0], batch[1], batch[2], dec_domains, dec_contents)
                            data.append(transfers)
                        data_save_dir = f"data/variants/RMNIST_augmented/RMNIST_train_{domain_string}/{domain}/{content}/data.pt"
                        os.makedirs(data_save_dir[:-8], exist_ok=True)
                        data = torch.cat(data, dim=0)
                        print(data.shape)
                        torch.save(data, data_save_dir)

            model.train()



