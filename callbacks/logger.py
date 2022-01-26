import os
from pytorch_lightning.callbacks import Callback
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


class Logger(Callback):
    def __init__(self, output_dir, train_batch, val_batch, domains, contents, images_on_val=False):
        super().__init__()
        self.output_dir = output_dir

        self.train_batch = train_batch
        self.val_batch = val_batch

        self.ave_grad_list = [[], [], [], [], [], [], [], [], [], []]
        self.max_grad_list = [[], [], [], [], [], [], [], [], [], []]

        self.train_loss = []
        self.val_loss = []

        self.domains = domains
        self.domain_dict = {domain: torch.LongTensor([i]) for i, domain in enumerate(self.domains)}
        self.contents = contents
        self.content_dict = {content: torch.LongTensor([i]) for i, content in enumerate(self.contents)}


        self.iov_flag = False
        self.images_on_val = images_on_val
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        os.makedirs(f"{self.output_dir}/version_{trainer.logger.version}/images", exist_ok=True)
        self.log_reconstructions(trainer, pl_module, tensorboard_log=True)
        self.log_generated(trainer, pl_module, tensorboard_log=True)
        self.log_losses(trainer)
        self.log_grad_flow(trainer, tensorboard_log=True)

        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        self.gather_grad_flow_data(pl_module)
        self.train_loss.append(outputs["loss"].item())

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, unused)

    def on_epoch_end(self, trainer, pl_module):
        self.iov_flag = True
        return super().on_epoch_end(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_loss.append(outputs.item())
        if self.images_on_val and self.iov_flag and batch_idx==20:
            os.makedirs(f"{self.output_dir}/version_{trainer.logger.version}/images", exist_ok=True)
            self.log_reconstructions(trainer, pl_module, tensorboard_log=True)
            self.log_generated(trainer, pl_module, tensorboard_log=True)
            self.log_losses(trainer)
            self.log_grad_flow(trainer, tensorboard_log=True)

        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def log_reconstructions(self, trainer, pl_module, tensorboard_log=False):
        with torch.no_grad():
            pl_module.eval()

            train_imgs = self.train_batch[0].to(pl_module.device)
            train_domains = self.train_batch[1].to(pl_module.device)
            train_contents = self.train_batch[2].to(pl_module.device)
            train_recs = pl_module.reconstruct(train_imgs, train_domains, train_contents)
            train_imgs = (train_imgs + 1.0) / 2.0
            train_recs = (train_recs + 1.0) / 2.0
            train_grid = torchvision.utils.make_grid(torch.stack((train_imgs, train_recs), dim=1).view(-1, 3, 224, 224))
            torchvision.utils.save_image(train_grid, f"{self.output_dir}/version_{trainer.logger.version}/images/train_reconstructions.png")

            val_imgs = self.val_batch[0][:max(8, len(self.val_batch[0]))].to(pl_module.device)
            val_domains = self.val_batch[1][:max(8, len(self.val_batch[0]))].to(pl_module.device)
            val_contents = self.val_batch[2][:max(8, len(self.val_batch[0]))].to(pl_module.device)
            val_recs = pl_module.reconstruct(val_imgs, val_domains, val_contents)
            val_imgs = (val_imgs + 1.0) / 2.0
            val_recs = (val_recs + 1.0) / 2.0
            val_grid = torchvision.utils.make_grid(torch.stack((val_imgs, val_recs), dim=1).view(-1, 3, 224, 224))
            torchvision.utils.save_image(val_grid, f"{self.output_dir}/version_{trainer.logger.version}/images/val_reconstructions.png")

            if tensorboard_log:
                trainer.logger.experiment.add_image("train_reconstruction", train_grid)
                trainer.logger.experiment.add_image("val_reconstruction", val_grid)

            pl_module.train()

    def log_grad_flow(self, trainer, tensorboard_log=False):
        """
        Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
        """
        fig = plt.figure(figsize=(24, 16))
        for mg in self.max_grad_list:
            plt.bar(np.arange(len(mg)), mg, alpha=0.1, lw=1, color="c")
        for ag in self.ave_grad_list:
            plt.bar(np.arange(len(ag)), ag, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(self.ave_grad_list[-1])+1, lw=2, color="k" )
        plt.xticks(range(0,len(self.ave_grad_list[-1]), 1), self.layers, rotation=45)
        plt.xlim(left=0, right=len(self.ave_grad_list[-1]))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig(f"{self.output_dir}/version_{trainer.logger.version}/images/gradient_flow.png")
        if tensorboard_log:
            trainer.logger.experiment.add_figure("gradient_flow", fig, close=False)
        plt.close(fig)
        fig = plt.figure(figsize=(24, 16))
        for mg in self.max_grad_list:
            plt.bar(np.arange(len(mg)), mg, alpha=0.1, lw=1, color="c")
        for ag in self.ave_grad_list:
            plt.bar(np.arange(len(ag)), ag, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(self.ave_grad_list[-1])+1, lw=2, color="k" )
        plt.xticks(range(0,len(self.ave_grad_list[-1]), 1), self.layers, rotation=45)
        plt.xlim(left=0, right=len(self.ave_grad_list[-1]))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig(f"{self.output_dir}/version_{trainer.logger.version}/images/gradient_flow_zoomed.png")
        if tensorboard_log:
            trainer.logger.experiment.add_figure("gradient_flow_zoomed", fig, close=False)
        plt.close(fig)

    def gather_grad_flow_data(self, pl_module):
        layers = []
        ave_grads = []
        max_grads = []
        for n, p in pl_module.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                if p.grad is not None:
                    ave_grads.append(p.grad.abs().mean().cpu())
                    max_grads.append(p.grad.abs().max().cpu())
                else:
                    ave_grads.append(0)
                    max_grads.append(0)
        self.layers = layers
        self.ave_grad_list.pop(0)
        self.ave_grad_list.append(ave_grads)
        self.max_grad_list.pop(0)
        self.max_grad_list.append(max_grads)

    def log_losses(self, trainer):
        plt.figure(figsize=(16, 8))
        plt.plot(self.train_loss)
        plt.title("training loss", size=18)
        plt.savefig(f"{self.output_dir}/version_{trainer.logger.version}/images/train_loss.png")
        plt.close()
        plt.figure(figsize=(16, 8))
        plt.plot(self.val_loss)
        plt.title("validation loss", size=18)
        plt.savefig(f"{self.output_dir}/version_{trainer.logger.version}/images/val_loss.png")
        plt.close()

    def log_generated(self, trainer, pl_module, tensorboard_log=False):
        with torch.no_grad():
            pl_module.eval()
            for domain_name in self.domains:
                for content_name in self.contents:
                    domains = torch.nn.functional.one_hot(self.domain_dict[domain_name], num_classes=len(self.domains)).repeat(self.train_batch[0].shape[0], 1).to(pl_module.device)
                    contents = torch.nn.functional.one_hot(self.content_dict[content_name], num_classes=len(self.contents)).repeat(self.train_batch[0].shape[0], 1).to(pl_module.device)
                    codes = torch.randn(size=(self.train_batch[0].shape[0], pl_module.latent_size)).to(pl_module.device)
                    reconstructions = pl_module.decoder(codes, domains, contents)
                    reconstructions = (reconstructions + 1.0) / 2.0
                    gen_grid = torchvision.utils.make_grid(reconstructions)
                    torchvision.utils.save_image(gen_grid, f"{self.output_dir}/version_{trainer.logger.version}/images/generated_{domain_name}_{content_name}.png")

                    if tensorboard_log:
                        trainer.logger.experiment.add_image(f"generated_{domain_name}_{content_name}", gen_grid)

            pl_module.train()