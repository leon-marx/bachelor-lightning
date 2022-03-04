from argparse import ArgumentParser
import os
from tkinter import Image
import pytorch_lightning as pl
import torch
import copy
import random
import torchvision

# Own Modules
from datasets.pacs_balanced import BalancedPACSDataModule
from models.cvae_v3 import CVAE_v3


def shift(image):
    image = (image + 1.0) / 2.0
    return torch.clamp(image, 0.0, 1.0)


def latent_neighbourhood(model, train_batch, val_batch, version_path, num_channels, image_size):
    model.eval()
    train_imgs = train_batch[0].to(model.device)
    train_domains = train_batch[1].to(model.device)
    train_contents = train_batch[2].to(model.device)

    train_enc_mu, train_enc_logvar = model.encoder(train_imgs, train_domains, train_contents)
    train_noise = torch.normal(mean=torch.zeros_like(train_enc_mu), std=torch.ones_like(train_enc_mu) * 5).to(model.device)
    train_z_list = [
        train_enc_mu,
        train_enc_mu + torch.randn_like(train_enc_mu) * (0.5 * train_enc_logvar).exp(),
        train_enc_mu + train_noise * (0.5 * train_enc_logvar).exp() * 0.1,
        train_enc_mu + train_noise * (0.5 * train_enc_logvar).exp() * 0.2,
        train_enc_mu + train_noise * (0.5 * train_enc_logvar).exp() * 0.5,
        train_enc_mu + train_noise * (0.5 * train_enc_logvar).exp(),
        train_enc_mu + torch.ones_like(train_enc_mu) * (0.5 * train_enc_logvar).exp() * 0.1,
        train_enc_mu + torch.ones_like(train_enc_mu) * (0.5 * train_enc_logvar).exp() * 0.5,
        train_enc_mu + torch.ones_like(train_enc_mu) * (0.5 * train_enc_logvar).exp(),
    ]
    train_reconstructions = tuple()
    for z in train_z_list:
        train_reconstructions += (shift(model.decoder(z, train_domains, train_contents)),)
    train_imgs = shift(train_imgs)
    train_grid = torchvision.utils.make_grid(torch.stack((train_imgs,) + train_reconstructions, dim=1).view(-1, num_channels, image_size, image_size))
    torchvision.utils.save_image(train_grid, f"{version_path}images/train_latent_neighbourhood.png")

    val_imgs = val_batch[0][:max(8, len(val_batch[0]))].to(model.device)
    val_domains = val_batch[1][:max(8, len(val_batch[0]))].to(model.device)
    val_contents = val_batch[2][:max(8, len(val_batch[0]))].to(model.device)

    val_enc_mu, val_enc_logvar = model.encoder(val_imgs, val_domains, val_contents)
    val_noise = torch.normal(mean=torch.zeros_like(val_enc_mu), std=torch.ones_like(val_enc_mu) * 5).to(model.device)
    val_z_list = [
        val_enc_mu,
        val_enc_mu + torch.randn_like(val_enc_mu) * (0.5 * val_enc_logvar).exp(),
        val_enc_mu + val_noise * (0.5 * val_enc_logvar).exp() * 0.1,
        val_enc_mu + val_noise * (0.5 * val_enc_logvar).exp() * 0.2,
        val_enc_mu + val_noise * (0.5 * val_enc_logvar).exp() * 0.5,
        val_enc_mu + val_noise * (0.5 * val_enc_logvar).exp(),
        val_enc_mu + torch.ones_like(val_enc_mu) * (0.5 * val_enc_logvar).exp() * 0.1,
        val_enc_mu + torch.ones_like(val_enc_mu) * (0.5 * val_enc_logvar).exp() * 0.5,
        val_enc_mu + torch.ones_like(val_enc_mu) * (0.5 * val_enc_logvar).exp(),
    ]
    val_reconstructions = tuple()
    for z in val_z_list:
        val_reconstructions += (shift(model.decoder(z, val_domains, val_contents)),)
    val_imgs = shift(val_imgs)
    val_grid = torchvision.utils.make_grid(torch.stack((val_imgs,) + val_reconstructions, dim=1).view(-1, num_channels, image_size, image_size))
    torchvision.utils.save_image(val_grid, f"{version_path}images/images/val_latent_neighbourhood.png")

    model.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()
    ####################
    # Configuration
    pl.seed_everything(17, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Printing Configuration
    print("Environment:")
    print(f"    PyTorch: {torch.__version__}")
    print(f"    CUDA: {torch.version.cuda}")
    print(f"    CUDNN: {torch.backends.cudnn.version()}")

    # Dataset
    batch_size = 2
    domains = ["art_painting", "cartoon", "photo"]
    contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
    dm = BalancedPACSDataModule(root="data", domains=domains, contents=contents,
                        batch_size=batch_size, num_workers=20)
    log_dm = BalancedPACSDataModule(root="data", domains=domains, contents=contents,
                        batch_size=batch_size, num_workers=20, shuffle_all=True)
    log_dm.setup()
    train_batch = next(iter(log_dm.train_dataloader()))
    val_batch = next(iter(log_dm.val_dataloader()))

    model = CVAE_v3.load_from_checkpoint(
        args.ckpt_path,
        data="PACS",
        num_domains=3,
        num_contents=7,
        latent_size=1024,
        lr=1e-5,
        depth=1,
        out_channels=[256, 256, 512, 512, 1024, 1024],
        kernel_size=3,
        activation=torch.nn.ELU(),
        downsampling="maxpool",
        upsampling="upsample",
        dropout=False,
        batch_norm=True,
        loss_mode="elbo",
        lamb=0,
        initialize=False,
        max_lamb=0
        )

    version_path = ""
    version_path_pieces = args.ckpt_path.split("/")
    for piece in version_path_pieces[:-1]:
        version_path += piece + "/"

    num_channels = 3
    image_size = 224

    latent_neighbourhood(model, train_batch, val_batch, version_path, num_channels, image_size)
