import os
from pytorch_lightning.callbacks import Callback
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


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

            train_imgs = self.train_batch[0][:max(8, len(self.train_batch[0]))].to(pl_module.device)
            train_domains = self.train_batch[1][:max(8, len(self.train_batch[0]))].to(pl_module.device)
            train_contents = self.train_batch[2][:max(8, len(self.train_batch[0]))].to(pl_module.device)
            if pl_module.__class__.__name__ == "CVAE":
                train_recs = pl_module(train_imgs, train_domains, train_contents, raw=True)[2]
            if pl_module.__class__.__name__ == "AE":
                train_recs = pl_module(train_imgs, train_domains, train_contents)
            train_grid = torchvision.utils.make_grid(torch.stack((train_imgs, train_recs), dim=1).view(-1, 3, 224, 224))
            torchvision.utils.save_image(train_grid, f"{self.out_dir}/images/train_reconstructions.png")

            val_imgs = self.val_batch[0][:max(8, len(self.val_batch[0]))].to(pl_module.device)
            val_domains = self.val_batch[1][:max(8, len(self.val_batch[0]))].to(pl_module.device)
            val_contents = self.val_batch[2][:max(8, len(self.val_batch[0]))].to(pl_module.device)
            if pl_module.__class__.__name__ == "CVAE":
                val_recs = pl_module(val_imgs, val_domains, val_contents, raw=True)[2]
            if pl_module.__class__.__name__ == "AE":
                val_recs = pl_module(val_imgs, val_domains, val_contents)
            val_grid = torchvision.utils.make_grid(torch.stack((val_imgs, val_recs), dim=1).view(-1, 3, 224, 224))
            torchvision.utils.save_image(val_grid, f"{self.out_dir}/images/val_reconstructions.png")

            pl_module.train()

    def log_grad_flow(self, trainer, pl_module, checkpoint):
        """
        Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
        """
        layers = []
        ave_grads = []
        max_grads = []
        for n, p in pl_module.named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu())
                max_grads.append(p.grad.abs().max().cpu())
        self.ave_grad_list.pop(0)
        self.ave_grad_list.append(ave_grads)
        self.max_grad_list.pop(0)
        self.max_grad_list.append(max_grads)
        plt.figure(figsize=(24, 16))
        for mg in self.max_grad_list:
            plt.bar(np.arange(len(mg)), mg, alpha=0.1, lw=1, color="c")
        for ag in self.ave_grad_list:
            plt.bar(np.arange(len(ag)), ag, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation=45)
        plt.xlim(left=0, right=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig(os.path.join(self.out_dir, "gradient_flow.png"))
        plt.close()
        plt.figure(figsize=(24, 16))
        for mg in self.max_grad_list:
            plt.bar(np.arange(len(mg)), mg, alpha=0.1, lw=1, color="c")
        for ag in self.ave_grad_list:
            plt.bar(np.arange(len(ag)), ag, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation=45)
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig(os.path.join(self.out_dir, "gradient_flow_zoomed.png"))
        plt.close()
