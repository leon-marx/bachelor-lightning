import os
from pytorch_lightning.callbacks import Callback
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm
import umap
plt.gray()

def shift(image):
    image = (image + 1.0) / 2.0
    return torch.clamp(image, 0.0, 1.0)


class ClassificationLogger(Callback):
    def __init__(self, output_dir, log_dm, train_batch, val_batch, domains, contents, images_on_val=False):
        super().__init__()
        self.output_dir = output_dir

        self.log_dm = log_dm

        self.train_batch = train_batch
        self.val_batch = val_batch
        self.num_channels = self.train_batch[0].shape[1]
        self.image_size = self.train_batch[0].shape[2]

        self.ave_grad_list = [[], [], [], [], [], [], [], [], [], []]
        self.max_grad_list = [[], [], [], [], [], [], [], [], [], []]

        self.train_loss = []
        self.val_loss = []

        self.domains = domains
        self.domain_dict = {domain: torch.LongTensor([i]) for i, domain in enumerate(self.domains)}
        self.contents = contents
        self.content_dict = {content: torch.LongTensor([i]) for i, content in enumerate(self.contents)}

        self.epoch_counter = -1
        self.umap_freq = 10
        self.iov_flag = False
        self.images_on_val = images_on_val

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        os.makedirs(f"{self.output_dir}/version_{trainer.logger.version}/images", exist_ok=True)
        self.log_classifications(trainer, pl_module, tensorboard_log=True)
        self.log_losses(trainer)
        self.log_grad_flow(trainer, tensorboard_log=True)

        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        self.gather_grad_flow_data(pl_module)
        self.train_loss.append(outputs["loss"].item())

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, unused)

    def on_epoch_end(self, trainer, pl_module):
        self.iov_flag = True
        self.epoch_counter += 1
        if self.epoch_counter / 2 >= self.umap_freq:
            self.log_umap(trainer, pl_module)
            self.epoch_counter = 0
        return super().on_epoch_end(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_loss.append(outputs.item())
        if self.images_on_val and self.iov_flag and batch_idx==0:
            os.makedirs(f"{self.output_dir}/version_{trainer.logger.version}/images", exist_ok=True)
            self.log_classifications(trainer, pl_module, tensorboard_log=True)
            self.log_losses(trainer)
            self.log_grad_flow(trainer, tensorboard_log=True)

        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def log_classifications(self, trainer, pl_module, tensorboard_log=False):
        with torch.no_grad():
            pl_module.eval()

            train_imgs = self.train_batch[0].to(pl_module.device)
            train_contents = self.train_batch[2].to(pl_module.device)
            train_preds = pl_module(train_imgs)
            train_imgs = shift(train_imgs)

            fig = plt.figure(figsize=(10, 10))
            for i in range(min(train_imgs.shape[0], 16)):
                plt.subplot(4, 4, i+1)
                plt.imshow(train_imgs[i].view(28, 28).cpu().numpy())
                plt.xticks([])
                plt.yticks([])
                plt.title(f"Pred: {torch.argmax(train_preds[i]).item()}, Truth: {torch.argmax(train_contents[i]).item()}")
            plt.savefig(f"{self.output_dir}/version_{trainer.logger.version}/images/train_classification.png")
            if tensorboard_log:
                trainer.logger.experiment.add_figure("train_classification", fig, close=False)
            plt.close(fig)

            val_imgs = self.val_batch[0].to(pl_module.device)
            val_contents = self.val_batch[2].to(pl_module.device)
            val_preds = pl_module(val_imgs)
            val_imgs = shift(val_imgs)

            fig = plt.figure(figsize=(10, 10))
            for i in range(min(val_imgs.shape[0], 16)):
                plt.subplot(4, 4, i+1)
                plt.imshow(val_imgs[i].view(28, 28).cpu().numpy())
                plt.xticks([])
                plt.yticks([])
                plt.title(f"Pred: {torch.argmax(val_preds[i]).item()}, Truth: {torch.argmax(val_contents[i]).item()}")
            plt.savefig(f"{self.output_dir}/version_{trainer.logger.version}/images/val_classification.png")
            if tensorboard_log:
                trainer.logger.experiment.add_figure("val_classification", fig, close=False)
            plt.close(fig)

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
                    Line2D([0], [0], color="k", lw=4)], ["max-gradient", "mean-gradient", "zero-gradient"])
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
                    Line2D([0], [0], color="k", lw=4)], ["max-gradient", "mean-gradient", "zero-gradient"])
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

    def log_umap(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            first_dim = min(100, int(len(self.log_dm.train_dataloader())-1))
            latent_data = torch.zeros(size=(first_dim, self.train_batch[0].shape[0], 10))
            latent_domains = torch.zeros(size=(first_dim, self.train_batch[0].shape[0]))
            latent_contents = torch.zeros(size=(first_dim, self.train_batch[0].shape[0]))
            for i, batch in enumerate(iter(self.log_dm.train_dataloader())):
                images = batch[0].to(pl_module.device)
                domains = batch[1].to(pl_module.device)
                contents = batch[2].to(pl_module.device)
                latent_data[i] = pl_module(images).cpu()
                latent_domains[i] = torch.argmax(domains.cpu(), dim=1)
                latent_contents[i] = torch.argmax(contents.cpu(), dim=1)
                if i+1 >= first_dim:
                    break
            latent_data = latent_data.view(-1, 10)
            latent_domains = latent_domains.view(-1, 1)
            latent_contents = latent_contents.view(-1, 1)
            reducer = umap.UMAP(random_state=17)
            reducer.fit(latent_data)
            embedding = reducer.embedding_

            fig = plt.figure(figsize=(10, 8))
            plt.scatter(embedding[:, 0], embedding[:, 1], c=latent_domains, cmap="jet", s=5)
            plt.gca().set_aspect("equal", "datalim")
            cbar = plt.colorbar(boundaries=np.arange(len(self.domains)+1)-0.5)
            cbar.set_ticks(np.arange(len(self.domains)))
            cbar.ax.set_yticklabels(list(self.domain_dict.keys()))
            plt.title("UMAP projection of the latent space by domain", fontsize=14)
            plt.savefig(f"{self.output_dir}/version_{trainer.logger.version}/images/umap_by_domain.png")
            trainer.logger.experiment.add_figure("umap_by_domain", fig, close=False)
            plt.close(fig)
            fig = plt.figure(figsize=(10, 8))
            plt.scatter(embedding[:, 0], embedding[:, 1], c=latent_contents, cmap="jet", s=5)
            plt.gca().set_aspect("equal", "datalim")
            cbar = plt.colorbar(boundaries=np.arange(len(self.contents)+1)-0.5)
            cbar.set_ticks(np.arange(len(self.contents)))
            cbar.ax.set_yticklabels(list(self.content_dict.keys()))
            plt.title("UMAP projection of the latent space by content", fontsize=14)
            plt.savefig(f"{self.output_dir}/version_{trainer.logger.version}/images/umap_by_content.png")
            trainer.logger.experiment.add_figure("umap_by_content", fig, close=False)
            plt.close(fig)
            pl_module.train()
