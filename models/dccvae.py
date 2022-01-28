import torch
import pytorch_lightning as pl


def dc_init(m):
    """
    Initialization suggested by authors of DCGAN paper.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


class DCCVAE(pl.LightningModule):
    def __init__(self, num_domains, num_contents, latent_size, feature_size, lr, loss_mode, lamb):
        super().__init__()

        self.num_domains = num_domains
        self.num_contents = num_contents
        self.latent_size = latent_size
        self.feature_size = feature_size
        self.lr = lr
        self.loss_mode = loss_mode
        self.lamb = lamb
        self.hyper_param_dict = {
            "num_domains": self.num_domains,
            "num_contents": self.num_contents,
            "latent_size": latent_size,
            "feature_size": feature_size,
            "loss_mode": self.loss_mode,
            "lamb": self.lamb,
        }

        self.encoder = Encoder(num_domains=self.num_domains,
                               num_contents=self.num_contents,
                               latent_size=self.latent_size,
                               feature_size=self.feature_size
                               )
        self.decoder = Decoder(num_domains=self.num_domains,
                               num_contents=self.num_contents,
                               latent_size=self.latent_size,
                               feature_size=self.feature_size
                               )
        self.apply(dc_init)

    def loss(self, images, enc_mu, enc_logvar, reconstructions, split_loss=False):
        """
        Calculates the loss. Choose from l1, l2 and elbo

        images: Tensor of shape (batch_size, channels, height, width)
        enc_mu: Tensor of shape (batch_size, latent_size)
        enc_logvar: Tensor of shape (batch_size, latent_size)
        reconstructions: Tensor of shape (batch_size, channels, height, width)
        split_loss: bool, if True, returns kld and rec losses separately
        """
        if self.loss_mode == "l1":
            loss = torch.abs(images - reconstructions)
            return loss.mean(dim=[0, 1, 2, 3])
        if self.loss_mode == "l2":
            loss = torch.nn.functional.mse_loss(
                images, reconstructions, reduction="none")
            return loss.mean(dim=[0, 1, 2, 3])
        if self.loss_mode == "elbo":
            kld = self.lamb * 0.5 * (enc_mu ** 2 + enc_logvar.exp() - enc_logvar - 1).mean(dim=[0, 1])
            rec = torch.nn.functional.mse_loss(images, reconstructions, reduction="none").mean(dim=[0, 1, 2, 3])
            if split_loss:
                return kld + rec, kld.item(), rec.item()
            else:
                return kld + rec

    def forward(self, images, domains, contents):
        """
        Calculates codes for the given images and returns their reconstructions.

        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        enc_mu, enc_logvar = self.encoder(images, domains, contents)
        codes = enc_mu + torch.randn_like(enc_mu) * (0.5 * enc_logvar).exp()
        reconstructions = self.decoder(codes, domains, contents)

        return enc_mu, enc_logvar, reconstructions

    def training_step(self, batch, batch_idx):
        """
        Calculates the chosen Loss.

        batch: List [x, domain, content, filenames]
            images: Tensor of shape (batch_size, channels, height, width)
            domains: Tensor of shape (batch_size, num_domains)
            contents: Tensor of shape (batch_size, num_contents)
            filenames: Tuple of strings of the form: {domain}/{content}/{fname}
        batch_idx: The index of the batch, not used.
        """
        images = batch[0]
        domains = batch[1]
        contents = batch[2]

        enc_mu, enc_logvar, reconstructions = self(images, domains, contents)

        loss, kld_value, rec_value = self.loss(
            images, enc_mu, enc_logvar, reconstructions, split_loss=True)
        self.log("train_loss", loss, batch_size=images.shape[0])
        self.log("kld", kld_value, prog_bar=True, batch_size=images.shape[0])
        self.log("rec", rec_value, prog_bar=True, batch_size=images.shape[0])
        # self.log("lr", self.optimizers(
        # ).param_groups[0]["lr"], prog_bar=True, batch_size=images.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Calculates the chosen Loss.

        batch: List [x, domain, content, filenames]
            images: Tensor of shape (batch_size, channels, height, width)
            domains: Tensor of shape (batch_size, num_domains)
            contents: Tensor of shape (batch_size, num_contents)
            filenames: Tuple of strings of the form: {domain}/{content}/{fname}
        batch_idx: The index of the batch, not used.
        """
        images = batch[0]
        domains = batch[1]
        contents = batch[2]

        enc_mu, enc_logvar, reconstructions = self(images, domains, contents)

        loss = self.loss(images, enc_mu, enc_logvar, reconstructions)
        self.log("val_loss", loss, batch_size=images.shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        """
        Calculates the chosen Loss.

        batch: List [x, domain, content, filenames]
            images: Tensor of shape (batch_size, channels, height, width)
            domains: Tensor of shape (batch_size, num_domains)
            contents: Tensor of shape (batch_size, num_contents)
            filenames: Tuple of strings of the form: {domain}/{content}/{fname}
        batch_idx: The index of the batch, not used.
        """
        images = batch[0]
        domains = batch[1]
        contents = batch[2]

        enc_mu, enc_logvar, reconstructions = self(images, domains, contents)

        return self.loss(images, enc_mu, enc_logvar, reconstructions)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return optimizer

    def reconstruct(self, images, domains, contents):
        """
        Calculates codes for the given images and returns their reconstructions.

        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        enc_mu, enc_logvar = self.encoder(images, domains, contents)
        reconstructions = self.decoder(enc_mu, domains, contents)

        return reconstructions

    def generate(self, codes, domains, contents):
        """
        Generate images from Gaussian distributed codes.
        """
        with torch.no_grad():
            self.eval()
            x = torch.cat((codes, domains, contents), dim=1)
            x = self.decoder.reshape(x)
            reconstructions = self.decoder.dec_conv_sequential(x)
            self.train()
            return reconstructions

class Encoder(torch.nn.Module):
    def __init__(self, num_domains, num_contents, latent_size, feature_size):
        super().__init__()
        self.num_domains = num_domains
        self.num_contents = num_contents
        self.latent_size = latent_size
        self.feature_size = feature_size
        self.enc_conv_sequential = torch.nn.Sequential(
            *self.block(
                in_channels=3 + self.num_domains + self.num_contents,
                out_channels=self.feature_size,
            ),  # (N, fs, 112, 112)
            *self.block(
                in_channels=self.feature_size,
                out_channels=self.feature_size * 2,
            ),  # (N, 2fs, 56, 56)
            *self.block(
                in_channels=self.feature_size * 2,
                out_channels=self.feature_size * 4,
            ),  # (N, 4fs, 28, 28)
            *self.block(
                in_channels=self.feature_size * 4,
                out_channels=self.feature_size * 8,
            ),  # (N, 8fs, 14, 14)
            *self.block(
                in_channels=self.feature_size * 8,
                out_channels=self.feature_size * 16,
            ),  # (N, 16fs, 7, 7)
            *self.block(
                in_channels=self.feature_size * 16,
                out_channels=self.feature_size * 32,
            ),  # (N, 32fs, 3, 3)
        )
        self.get_mu = torch.nn.Sequential(
            *self.block(
                in_channels=self.feature_size * 32,
                out_channels=self.latent_size,
                last=True
            ),  # (N, latent_size, 1, 1)
            torch.nn.Flatten()  # (N, latent_size)
        )
        self.get_logvar = torch.nn.Sequential(
            *self.block(
                in_channels=self.feature_size * 32,
                out_channels=self.latent_size,
                last=True
            ),  # (N, latent_size, 1, 1)
            torch.nn.Flatten()  # (N, latent_size)
        )

    def block(self, in_channels, out_channels, last=False):
        if last:
            seq = [
                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                stride=1, padding=0, bias=False),
                torch.nn.Sigmoid(),
            ]
        else:
            seq = [
                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4,
                                stride=2, padding=1, bias=False),
                torch.nn.BatchNorm2d(num_features=out_channels),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ]
        return seq

    def forward(self, images, domains, contents):
        """
        Calculates latent-space encodings for the given images in the form p(z | x).

        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        domain_panels = torch.ones(size=(images.shape[0], self.num_domains, 224, 224)).to(
            images.device) * domains.view(images.shape[0], self.num_domains, 1, 1)
        content_panels = torch.ones(size=(images.shape[0], self.num_contents, 224, 224)).to(
            images.device) * contents.view(images.shape[0], self.num_contents, 1, 1)

        x = torch.cat((images, domain_panels, content_panels), dim=1)
        x = self.enc_conv_sequential(x)
        enc_mu = self.get_mu(x)
        enc_logvar = self.get_logvar(x)

        return enc_mu, enc_logvar


class Decoder(torch.nn.Module):
    def __init__(self, num_domains, num_contents, latent_size, feature_size):
        super().__init__()
        self.num_domains = num_domains
        self.num_contents = num_contents
        self.latent_size = latent_size
        self.feature_size = feature_size
        self.reshape = lambda x: x.view(-1, self.latent_size + self.num_domains + self.num_contents, 1, 1)
        self.dec_conv_sequential = torch.nn.Sequential(
            *self.block(
                in_channels=self.latent_size + self.num_domains + self.num_contents,
                out_channels=self.feature_size * 32,
                first=True
            ),  # (N, 32fs, 3, 3)
            *self.block(
                in_channels=self.feature_size * 32,
                out_channels=self.feature_size * 16,
                three_to_seven=True
            ),  # (N, 16fs, 7, 7)
            *self.block(
                in_channels=self.feature_size * 16,
                out_channels=self.feature_size * 8,
            ),  # (N, 8fs, 14, 14)
            *self.block(
                in_channels=self.feature_size * 8,
                out_channels=self.feature_size * 4,
            ),  # (N, 4fs, 28, 28)
            *self.block(
                in_channels=self.feature_size * 4,
                out_channels=self.feature_size * 2,
            ),  # (N, 2fs, 56, 56)
            *self.block(
                in_channels=self.feature_size * 2,
                out_channels=self.feature_size,
            ),  # (N, fs, 112, 112)
            *self.block(
                in_channels=self.feature_size,
                out_channels=3,
                last=True
            ),  # (N, 3, 224, 224)
        )

    def block(self, in_channels, out_channels, first=False, three_to_seven=False, last=False):
        if first:
            seq = [
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                         stride=1, padding=0, bias=False),
                torch.nn.BatchNorm2d(num_features=out_channels),
                torch.nn.ReLU(inplace=True),
            ]
        elif three_to_seven:
            seq = [
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4,
                                         stride=2, padding=1, output_padding=1, bias=False),
                torch.nn.BatchNorm2d(num_features=out_channels),
                torch.nn.ReLU(inplace=True),
            ]
        elif last:
            seq = [
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4,
                                         stride=2, padding=1, bias=False),
                torch.nn.Tanh(),
            ]
        else:
            seq = [
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4,
                                         stride=2, padding=1, bias=False),
                torch.nn.BatchNorm2d(num_features=out_channels),
                torch.nn.ReLU(inplace=True),
            ]
        return seq

    def forward(self, codes, domains, contents):
        """
        Calculates reconstructions of the given latent-space encodings. 

        codes: Tensor of shape (batch_size, latent_size)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        x = torch.cat((codes, domains, contents), dim=1)
        x = self.reshape(x)
        reconstructions = self.dec_conv_sequential(x)
        return reconstructions


if __name__ == "__main__":
    batch_size = 4

    num_domains = 3
    num_contents = 7
    latent_size = 512
    feature_size = 32

    lr = 1e-4
    loss_mode = "elbo"
    lamb = 0.1

    batch = [
        torch.randn(size=(batch_size, 3, 224, 224)),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_domains, size=(batch_size,)), num_classes=num_domains),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_contents, size=(batch_size,)), num_classes=num_contents),
        (f"pic_{i}" for i in range(batch_size))
    ]
    model = DCCVAE(num_domains=num_domains, num_contents=num_contents, lr=lr,
                   latent_size=latent_size, feature_size=feature_size, loss_mode=loss_mode, lamb=lamb)
    loss = model.training_step(batch, 0)
    print(loss)
    print("Done!")
