import os
from pytorch_lightning.callbacks import Callback
import torch
import torchvision


class ImageLogger(Callback):
    def __init__(self, out_dir, train_batch, val_batch):
        super().__init__()
        self.out_dir = out_dir
        self.train_batch = train_batch
        self.val_batch = val_batch
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self.log_reconstructions(trainer, pl_module, checkpoint)
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def log_reconstructions(self, trainer, pl_module, checkpoint):
        with torch.no_grad():
            pl_module.eval()
            os.makedirs(f"{self.out_dir}/images", exist_ok=True)

            train_imgs = self.train_batch[0]
            train_domains = self.train_batch[1]
            train_contents = self.train_batch[2]
            train_recs = pl_module(train_imgs, train_domains, train_contents, raw=True)[2]
            train_grid = torchvision.utils.make_grid(torch.cat((train_imgs, train_recs), 0))
            torchvision.utils.save_image(train_grid, f"{self.out_dir}/images/train_reconstructions.png")

            val_imgs = self.val_batch[0]
            val_domains = self.val_batch[1]
            val_contents = self.val_batch[2]
            val_recs = pl_module(val_imgs, val_domains, val_contents, raw=True)[2]
            val_grid = torchvision.utils.make_grid(torch.cat((val_imgs, val_recs), 0))
            torchvision.utils.save_image(val_grid, f"{self.out_dir}/images/val_reconstructions.png")

            pl_module.train()
