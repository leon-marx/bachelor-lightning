import os
from pytorch_lightning.callbacks import Callback
import torch
import torchvision


class ImageLogger(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        self.train_batch = batch
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, unused)
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_batch = batch
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        with torch.no_grad():
            pl_module.eval()

            train_imgs = self.train_batch[0]
            train_domains = self.train_batch[1]
            train_contents = self.train_batch[2]
            train_recs = pl_module(train_imgs, train_domains, train_contents, raw=True)[2]
            train_grid = torchvision.utils.make_grid(torch.cat((train_imgs, train_recs), 0))
            pl_module.logger.experiment.add_image("train_images", train_grid, 0)

            val_imgs = self.val_batch[0]
            val_domains = self.val_batch[1]
            val_contents = self.val_batch[2]
            val_recs = pl_module(val_imgs, val_domains, val_contents, raw=True)[2]
            val_grid = torchvision.utils.make_grid(torch.cat((val_imgs, val_recs), 0))
            pl_module.logger.experiment.add_image("val_images", val_grid, 0)

            pl_module.train()
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

        

    #####################################################
    # 1) DO THIS with 0. element of each domain/content folder
    # -> for grad plot: think of something
    # 2) lr scheduler