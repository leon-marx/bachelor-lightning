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

            train_imgs = self.train_batch[0].to(pl_module.device)
            train_domains = self.train_batch[1].to(pl_module.device)
            train_contents = self.train_batch[2].to(pl_module.device)
            train_recs = pl_module(train_imgs, train_domains, train_contents, raw=True)[2]
            train_grid = torchvision.utils.make_grid(torch.Tensor(list(zip(train_imgs, train_recs))).view(-1, 3, 224, 224))
            torchvision.utils.save_image(train_grid, f"{self.out_dir}/images/train_reconstructions.png")

            val_imgs = self.val_batch[0].to(pl_module.device)
            val_domains = self.val_batch[1].to(pl_module.device)
            val_contents = self.val_batch[2].to(pl_module.device)
            val_recs = pl_module(val_imgs, val_domains, val_contents, raw=True)[2]
            val_grid = torchvision.utils.make_grid(torch.Tensor(list(zip(val_imgs, val_recs))).view(-1, 3, 224, 224))
            torchvision.utils.save_image(val_grid, f"{self.out_dir}/images/val_reconstructions.png")

            pl_module.train()
