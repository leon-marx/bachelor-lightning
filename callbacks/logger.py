import os
from pytorch_lightning.callbacks import Callback
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm
import umap

from models.cvae_v3 import CVAE_v3
from models.mmd_cvae import MMD_CVAE

def shift(image):
    image = (image + 1.0) / 2.0
    return torch.clamp(image, 0.0, 1.0)


class Logger(Callback):
    def __init__(self, output_dir, log_dm, train_batch, val_batch, domains, contents, images_on_val=False):
        super().__init__()
        self.output_dir = output_dir

        self.log_dm = log_dm

        self.train_batch = train_batch
        self.val_batch = val_batch
        self.num_channels = self.train_batch[0].shape[1]
        self.image_size = self.train_batch[0].shape[2]
        self.batch_size = train_batch[0].shape[0]

        self.ave_grad_list = [[], [], [], [], [], [], [], [], [], []]
        self.max_grad_list = [[], [], [], [], [], [], [], [], [], []]

        self.train_loss = []
        self.val_loss = []

        self.domains = domains
        self.domain_dict = {domain: torch.LongTensor([i]) for i, domain in enumerate(self.domains)}
        self.domain_backwards_dict = {i: domain for i, domain in enumerate(self.domains)}
        self.contents = contents
        self.content_dict = {content: torch.LongTensor([i]) for i, content in enumerate(self.contents)}

        self.epoch_counter = -1
        self.warumup_freq = 5
        self.iov_flag = False
        self.images_on_val = images_on_val

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        os.makedirs(f"{self.output_dir}/version_{trainer.logger.version}/images", exist_ok=True)
        self.log_reconstructions(trainer, pl_module, tensorboard_log=True)
        self.latent_neighbourhood(trainer, pl_module, tensorboard_log=True)
        self.log_generated(trainer, pl_module, tensorboard_log=True)
        self.log_losses(trainer)
        self.log_grad_flow(trainer, tensorboard_log=True)

        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        self.gather_grad_flow_data(pl_module)
        if isinstance(outputs, list):
            self.train_loss.append(outputs[0]["loss"].item())
            if torch.isnan(outputs[0]["loss"]).item():
                print("NaN Loss, raising KeyboardInterrupt")
                raise KeyboardInterrupt
                # print("REDUCING LEARNING RATE!")
                # trainer.optimizer = pl_module.configure_optimizers(reduce_lr=True)
        else:
            self.train_loss.append(outputs["loss"].item())
            if torch.isnan(outputs["loss"]).item():
                print("NaN Loss, raising KeyboardInterrupt")
                raise KeyboardInterrupt
                # print("REDUCING LEARNING RATE!")
                # pl_module.optimizer = pl_module.configure_optimizers(reduce_lr=True)

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, unused)

    def on_epoch_end(self, trainer, pl_module):
        self.iov_flag = True
        self.epoch_counter += 1
        if self.epoch_counter / 2 >= self.warumup_freq:
            if not self.images_on_val:
                os.makedirs(f"{self.output_dir}/version_{trainer.logger.version}/images", exist_ok=True)
                self.log_reconstructions(trainer, pl_module, tensorboard_log=True)
                self.latent_neighbourhood(trainer, pl_module, tensorboard_log=True)
                self.log_generated(trainer, pl_module, tensorboard_log=True)
                self.log_domain_transfers(trainer, pl_module, tensorboard_log=True)
                self.log_content_transfers(trainer, pl_module, tensorboard_log=True)
                self.log_losses(trainer)
                self.log_grad_flow(trainer, tensorboard_log=True)
            self.log_umap(trainer, pl_module)
            if self.image_size == 28:
                self.check_overfit(trainer, pl_module)
            self.epoch_counter = 0
            if getattr(pl_module, "warmer", None) is not None:
                pl_module.warmer()
        return super().on_epoch_end(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_loss.append(outputs.item())
        if self.images_on_val and self.iov_flag and batch_idx==0:
            os.makedirs(f"{self.output_dir}/version_{trainer.logger.version}/images", exist_ok=True)
            self.log_reconstructions(trainer, pl_module, tensorboard_log=True)
            self.latent_neighbourhood(trainer, pl_module, tensorboard_log=True)
            self.log_generated(trainer, pl_module, tensorboard_log=True)
            self.log_domain_transfers(trainer, pl_module, tensorboard_log=True)
            self.log_content_transfers(trainer, pl_module, tensorboard_log=True)
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
            train_imgs = shift(train_imgs)
            train_recs = shift(train_recs)
            train_grid = torchvision.utils.make_grid(torch.stack((train_imgs, train_recs), dim=1).view(-1, self.num_channels, self.image_size, self.image_size))
            torchvision.utils.save_image(train_grid, f"{self.output_dir}/version_{trainer.logger.version}/images/train_reconstructions.png")

            val_imgs = self.val_batch[0][:max(8, len(self.val_batch[0]))].to(pl_module.device)
            val_domains = self.val_batch[1][:max(8, len(self.val_batch[0]))].to(pl_module.device)
            val_contents = self.val_batch[2][:max(8, len(self.val_batch[0]))].to(pl_module.device)
            val_recs = pl_module.reconstruct(val_imgs, val_domains, val_contents)
            val_imgs = shift(val_imgs)
            val_recs = shift(val_recs)
            val_grid = torchvision.utils.make_grid(torch.stack((val_imgs, val_recs), dim=1).view(-1, self.num_channels, self.image_size, self.image_size))
            torchvision.utils.save_image(val_grid, f"{self.output_dir}/version_{trainer.logger.version}/images/val_reconstructions.png")

            if tensorboard_log:
                trainer.logger.experiment.add_image("train_reconstruction", train_grid)
                trainer.logger.experiment.add_image("val_reconstruction", val_grid)

            pl_module.train()

    def log_domain_transfers(self, trainer, pl_module, tensorboard_log=False):
        with torch.no_grad():
            pl_module.eval()

            train_imgs = self.train_batch[0].to(pl_module.device)
            train_domains = self.train_batch[1].to(pl_module.device)
            train_contents = self.train_batch[2].to(pl_module.device)

            for decoder_domain in self.domains:
                dec_domains = torch.cat((torch.nn.functional.one_hot(self.domain_dict[decoder_domain], num_classes=len(self.domains)),) * train_domains.shape[0], dim=0).to(pl_module.device)

                transfers = pl_module.transfer(train_imgs, train_domains, train_contents, dec_domains, train_contents).cpu()
                train_imgs_to_plot = shift(train_imgs.cpu())
                transfers = shift(transfers)
                transfer_grid = torchvision.utils.make_grid(torch.stack((train_imgs_to_plot, transfers), dim=1).view(-1, self.num_channels, self.image_size, self.image_size))
                torchvision.utils.save_image(transfer_grid, f"{self.output_dir}/version_{trainer.logger.version}/images/domain_transfer_to_{decoder_domain}.png")

                if tensorboard_log:
                    trainer.logger.experiment.add_image(f"domain_transfer_to_{decoder_domain}", transfer_grid)

            pl_module.train()

    def log_content_transfers(self, trainer, pl_module, tensorboard_log=False):
        with torch.no_grad():
            pl_module.eval()

            train_imgs = self.train_batch[0].to(pl_module.device)
            train_domains = self.train_batch[1].to(pl_module.device)
            train_contents = self.train_batch[2].to(pl_module.device)

            for decoder_content in self.contents:
                dec_contents = torch.cat((torch.nn.functional.one_hot(self.content_dict[decoder_content], num_classes=len(self.contents)),) * train_contents.shape[0], dim=0).to(pl_module.device)

                transfers = pl_module.transfer(train_imgs, train_domains, train_contents, train_domains, dec_contents).cpu()
                train_imgs_to_plot = shift(train_imgs.cpu())
                transfers = shift(transfers)
                transfer_grid = torchvision.utils.make_grid(torch.stack((train_imgs_to_plot, transfers), dim=1).view(-1, self.num_channels, self.image_size, self.image_size))
                torchvision.utils.save_image(transfer_grid, f"{self.output_dir}/version_{trainer.logger.version}/images/content_transfer_to_{decoder_content}.png")

                if tensorboard_log:
                    trainer.logger.experiment.add_image(f"content_transfer_to_{decoder_content}", transfer_grid)

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

    def log_generated(self, trainer, pl_module, tensorboard_log=False):
        with torch.no_grad():
            pl_module.eval()
            codes = torch.randn(size=(min(self.train_batch[0].shape[0], 8), pl_module.latent_size)).to(pl_module.device)
            for domain_name in self.domains:
                # for content_name in {3: ["dog"], 1: [4]}[self.num_channels]:
                for content_name in self.contents:
                    domains = torch.nn.functional.one_hot(self.domain_dict[domain_name], num_classes=len(self.domains)).repeat(codes.shape[0], 1).to(pl_module.device)
                    contents = torch.nn.functional.one_hot(self.content_dict[content_name], num_classes=len(self.contents)).repeat(codes.shape[0], 1).to(pl_module.device)
                    reconstructions = pl_module.generate(codes, domains, contents)
                    reconstructions = shift(reconstructions)
                    gen_grid = torchvision.utils.make_grid(reconstructions)
                    torchvision.utils.save_image(gen_grid, f"{self.output_dir}/version_{trainer.logger.version}/images/generated_{domain_name}_{content_name}.png")

                    if tensorboard_log:
                        trainer.logger.experiment.add_image(f"generated_{domain_name}_{content_name}", gen_grid)

            pl_module.train()

    def log_umap(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            first_dim = min(50, int(len(self.log_dm.train_dataloader())-1))
            latent_data = torch.zeros(size=(first_dim, self.train_batch[0].shape[0], pl_module.latent_size))
            latent_domains = torch.zeros(size=(first_dim, self.train_batch[0].shape[0]))
            latent_contents = torch.zeros(size=(first_dim, self.train_batch[0].shape[0]))
            for i, batch in enumerate(iter(self.log_dm.train_dataloader())):
                images = batch[0].to(pl_module.device)
                domains = batch[1].to(pl_module.device)
                contents = batch[2].to(pl_module.device)
                if isinstance(pl_module, CVAE_v3) or isinstance(pl_module, MMD_CVAE):
                    enc_mu, enc_logvar = pl_module(images, domains, contents)[:2]
                    latent_data[i] = enc_mu + torch.randn_like(enc_mu) * (0.5 * enc_logvar).exp()
                else:
                    latent_data[i] = pl_module(images, domains, contents)[0].cpu()
                latent_domains[i] = torch.argmax(domains.cpu(), dim=1)
                latent_contents[i] = torch.argmax(contents.cpu(), dim=1)
                if i+1 >= latent_data.shape[0]:
                    break
            latent_data = latent_data.view(-1, pl_module.latent_size)
            latent_domains = latent_domains.view(-1, 1)
            latent_contents = latent_contents.view(-1, 1)
            reducer = umap.UMAP(random_state=17)
            reducer.fit(latent_data)
            embedding = reducer.embedding_
            fig = plt.figure(figsize=(10, 8))
            plt.scatter(latent_data[:, 0], latent_data[:, 1], c=latent_domains, cmap="jet", s=5)
            plt.gca().set_aspect("equal", "datalim")
            cbar = plt.colorbar(boundaries=np.arange(len(self.domains)+1)-0.5)
            cbar.set_ticks(np.arange(len(self.domains)))
            cbar.ax.set_yticklabels(list(self.domain_dict.keys()))
            plt.title("first 2 dimensions of the latent space by domain", fontsize=14)
            plt.savefig(f"{self.output_dir}/version_{trainer.logger.version}/images/first_two_by_domain.png")
            trainer.logger.experiment.add_figure("first_two_by_domain", fig, close=False)
            plt.close(fig)
            fig = plt.figure(figsize=(10, 8))
            plt.scatter(latent_data[:, 0], latent_data[:, 1], c=latent_contents, cmap="jet", s=5)
            plt.gca().set_aspect("equal", "datalim")
            cbar = plt.colorbar(boundaries=np.arange(len(self.contents)+1)-0.5)
            cbar.set_ticks(np.arange(len(self.contents)))
            cbar.ax.set_yticklabels(list(self.content_dict.keys()))
            plt.title("first 2 dimensions of the latent space by content", fontsize=14)
            plt.savefig(f"{self.output_dir}/version_{trainer.logger.version}/images/first_two_by_content.png")
            trainer.logger.experiment.add_figure("first_two_by_content", fig, close=False)
            plt.close(fig)
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

            fig = plt.figure(figsize=(10, 8))
            normal = torch.randn(size=(first_dim * self.train_batch[0].shape[0], pl_module.latent_size))
            normal_embedding = reducer.transform(normal)
            compare_embedding = np.concatenate((embedding, normal_embedding), axis=0)
            compare_colors = np.concatenate((np.zeros(normal_embedding.shape[0]), np.ones(normal_embedding.shape[0])), axis=0)
            plt.scatter(compare_embedding[:, 0], compare_embedding[:, 1], c=compare_colors, cmap="jet", s=5)
            plt.gca().set_aspect("equal", "datalim")
            cbar = plt.colorbar(boundaries=np.arange(2+1)-0.5)
            cbar.set_ticks(np.arange(2))
            cbar.ax.set_yticklabels(["latent space", "normal distribution"])
            plt.title("UMAP projection of the latent space and normal distribution", fontsize=14)
            plt.savefig(f"{self.output_dir}/version_{trainer.logger.version}/images/umap_normal.png")
            trainer.logger.experiment.add_figure("umap_normal", fig, close=False)
            plt.close(fig)
            pl_module.train()

    def check_overfit(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            bs = 4
            codes = torch.randn(size=(bs, pl_module.latent_size)).to(pl_module.device)
            generated_dict = {
                domain: {content: torch.zeros(size=(bs, 1, 28, 28)) for content in self.contents} for domain in self.domains
            }
            best_reconstruction_dict = {
                domain: {content: torch.zeros(size=(bs, 1, 28, 28)) for content in self.contents} for domain in self.domains
            }
            best_original_dict = {
                domain: {content: torch.zeros(size=(bs, 1, 28, 28)) for content in self.contents} for domain in self.domains
            }
            reconstruction_score_dict = {
                domain: {content: torch.ones(size=(bs,)) * 1000000 for content in self.contents} for domain in self.domains
            }
            original_score_dict = {
                domain: {content: torch.ones(size=(bs,)) * 1000000 for content in self.contents} for domain in self.domains
            }
            # generating all possible domain-content combinations (times 4)
            for domain_name in self.domains:
                for content_name in self.contents:
                    domains = torch.nn.functional.one_hot(self.domain_dict[domain_name], num_classes=len(self.domains)).repeat(codes.shape[0], 1).to(pl_module.device)
                    contents = torch.nn.functional.one_hot(self.content_dict[content_name], num_classes=len(self.contents)).repeat(codes.shape[0], 1).to(pl_module.device)
                    generated_images = pl_module.generate(codes, domains, contents)
                    generated_dict[domain_name][content_name] = generated_images

            # comparing whole training set with generated images
            for batch in tqdm(self.log_dm.train_dataloader()):
                images = batch[0].to(pl_module.device)
                domains = batch[1].to(pl_module.device)
                contents = batch[2].to(pl_module.device)
                reconstructions = pl_module.reconstruct(images, domains, contents)
                domains = torch.argmax(domains, dim=1)
                contents = torch.argmax(contents, dim=1)
                for i in range(images.shape[0]):
                    img = images[i]
                    dom = self.domain_backwards_dict[int(domains[i].item())]
                    cont = int(contents[i].item())
                    rec = reconstructions[i]
                    for j in range(bs):
                        rec_score = torch.nn.functional.mse_loss(generated_dict[dom][cont][j], rec)
                        if rec_score < reconstruction_score_dict[dom][cont][j]:
                            reconstruction_score_dict[dom][cont][j] = rec_score
                            best_reconstruction_dict[dom][cont][j] = rec
                        orig_score = torch.nn.functional.mse_loss(generated_dict[dom][cont][j], img)
                        if orig_score < original_score_dict[dom][cont][j]:
                            original_score_dict[dom][cont][j] = orig_score
                            best_original_dict[dom][cont][j] = img

            # making grids for each domain-content pair and all 4 generated images
            for dom in self.domains:
                for cont in self.contents:
                    canvas = torch.stack((
                        shift(generated_dict[dom][cont].cpu()),
                        shift(best_reconstruction_dict[dom][cont].cpu()),
                        shift(best_original_dict[dom][cont].cpu())),
                        dim=1).view(-1, 1, 28, 28)
                    gen_grid = torchvision.utils.make_grid(canvas, nrow=3)
                    torchvision.utils.save_image(gen_grid, f"{self.output_dir}/version_{trainer.logger.version}/images/generated_{dom}_{cont}_comparison.png")
                    trainer.logger.experiment.add_image(f"generated_{dom}_{cont}_comparison.png", gen_grid)

            pl_module.train()

    def latent_neighbourhood(self, trainer, pl_module, tensorboard_log=False):
        if False:  # We do not want to use this at the moment, this is a cheap flag in order to allow this easily
            with torch.no_grad():
                pl_module.eval()

                train_imgs = self.train_batch[0].to(pl_module.device)
                train_domains = self.train_batch[1].to(pl_module.device)
                train_contents = self.train_batch[2].to(pl_module.device)

                train_enc_mu, train_enc_logvar = pl_module.encoder(train_imgs, train_domains, train_contents)
                train_noise = torch.normal(mean=torch.zeros_like(train_enc_mu), std=torch.ones_like(train_enc_mu) * 5).to(pl_module.device)
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
                    train_reconstructions += (shift(pl_module.decoder(z, train_domains, train_contents)),)
                train_imgs = shift(train_imgs)
                train_grid = torchvision.utils.make_grid(torch.stack((train_imgs,) + train_reconstructions, dim=1).view(-1, self.num_channels, self.image_size, self.image_size))
                torchvision.utils.save_image(train_grid, f"{self.output_dir}/version_{trainer.logger.version}/images/train_latent_neighbourhood.png")

                val_imgs = self.val_batch[0][:max(8, len(self.val_batch[0]))].to(pl_module.device)
                val_domains = self.val_batch[1][:max(8, len(self.val_batch[0]))].to(pl_module.device)
                val_contents = self.val_batch[2][:max(8, len(self.val_batch[0]))].to(pl_module.device)

                val_enc_mu, val_enc_logvar = pl_module.encoder(val_imgs, val_domains, val_contents)
                val_noise = torch.normal(mean=torch.zeros_like(val_enc_mu), std=torch.ones_like(val_enc_mu) * 5).to(pl_module.device)
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
                    val_reconstructions += (shift(pl_module.decoder(z, val_domains, val_contents)),)
                val_imgs = shift(val_imgs)
                val_grid = torchvision.utils.make_grid(torch.stack((val_imgs,) + val_reconstructions, dim=1).view(-1, self.num_channels, self.image_size, self.image_size))
                torchvision.utils.save_image(val_grid, f"{self.output_dir}/version_{trainer.logger.version}/images/val_latent_neighbourhood.png")

                if tensorboard_log:
                    trainer.logger.experiment.add_image("train_latent_neighbourhood", train_grid)
                    trainer.logger.experiment.add_image("val_latent_neighbourhood", val_grid)

                pl_module.train()