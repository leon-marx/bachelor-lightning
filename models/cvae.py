import torch
import pytorch_lightning as pl


class CVAE(pl.LightningModule):
    def __init__(self, num_domains, num_contents, latent_size, lamb, lr):
        super().__init__()

        self.num_domains = num_domains
        self.num_contents = num_contents
        self.latent_size = latent_size

        self.encoder = Encoder(num_domains=self.num_domains, num_contents=self.num_contents, latent_size=self.latent_size)
        self.decoder = Decoder(num_domains=self.num_domains, num_contents=self.num_contents, latent_size=self.latent_size)

        self.flatten = torch.nn.Flatten()
        self.lamb = lamb
        self.lr = lr

    def loss(self, images, enc_mu, enc_logvar, dec_mu, dec_logvar, split_loss=False):
        """
        Calculates the ELBO Loss (negative ELBO).
        images: Tensor of shape (batch_size, channels, height, width)
        enc_mu: Tensor of shape (batch_size, latent_size)
        enc_logvar: Tensor of shape (batch_size, latent_size)
        dec_mu: Tensor of shape (batch_size, channels, height, width)
        dec_logvar: Tensor of shape (batch_size, channels, height, width)
        split_loss: If True, returns separate kld- and reconstruction-loss
        """
        images = self.flatten(images)
        dec_mu = self.flatten(dec_mu)
        dec_logvar = self.flatten(dec_logvar)
        kld = torch.mean((  # KL divergence -> regularization
            torch.sum((
                enc_mu ** 2 + enc_logvar.exp() - enc_logvar - \
                torch.ones(enc_mu.shape).to(images.device)
            ),
                dim=1) * 0.5 * (1.0 / self.lamb)
        ),
            dim=0)
        recon = torch.mean((  # likelihood -> similarity
            torch.sum((
                (images - dec_mu) ** 2 / (2 * dec_logvar.exp()) + 0.5 * dec_logvar
            ),
                dim=1)
        ),
            dim=0)

        if split_loss:
            return kld, recon
        else:
            return kld + recon

    def forward(self, images, domains, contents, raw=False):
        """
        Calculates mean and diagonal log-variance of p(z | x) and of p(x | z).

        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        raw: Bool, if True: z is sampled without noise
        """
        enc_mu, enc_logvar = self.encoder(images, domains, contents)

        if raw:
            codes = enc_mu
        else:
            z_std = (0.5 * enc_logvar).exp()
            z_eps = torch.randn_like(enc_mu)
            codes = enc_mu + z_eps * z_std

        dec_mu, dec_logvar = self.decoder(codes, domains, contents)

        return enc_mu, enc_logvar, dec_mu, dec_logvar

    def training_step(self, batch, batch_idx):
        """
        Calculates the ELBO Loss (negative ELBO).

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

        enc_mu, enc_logvar, dec_mu, dec_logvar = self(images, domains, contents)
        
        kld, rec = self.loss(images, enc_mu, enc_logvar, dec_mu, dec_logvar, split_loss=True)
        self.log("kld", kld, prog_bar=True)
        self.log("rec", rec, prog_bar=True)
        return kld + rec

    def validation_step(self, batch, batch_idx):
        """
        Calculates the ELBO Loss (negative ELBO).

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

        enc_mu, enc_logvar, dec_mu, dec_logvar = self(images, domains, contents)

        return self.loss(images, enc_mu, enc_logvar, dec_mu, dec_logvar)

    def test_step(self, batch, batch_idx):
        """
        Calculates the ELBO Loss (negative ELBO).

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

        enc_mu, enc_logvar, dec_mu, dec_logvar = self(images, domains, contents)

        return self.loss(images, enc_mu, enc_logvar, dec_mu, dec_logvar)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class Encoder(torch.nn.Module):
    def __init__(self, num_domains, num_contents, latent_size):
        super().__init__()
        self.num_domains = num_domains
        self.num_contents = num_contents
        self.latent_size = latent_size
        self.enc_conv_sequential = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3 + self.num_contents + self.num_domains,
                            out_channels=128, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.Conv2d(in_channels=128, out_channels=128,
                            kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.MaxPool2d(kernel_size=2),  # (N, 128, 112, 112)
            torch.nn.Conv2d(in_channels=128, out_channels=256,
                            kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.Conv2d(in_channels=256, out_channels=256,
                            kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.MaxPool2d(kernel_size=2),  # (N, 256, 56, 56)
            torch.nn.Conv2d(in_channels=256, out_channels=512,
                            kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.Conv2d(in_channels=512, out_channels=512,
                            kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.MaxPool2d(kernel_size=2),  # (N, 512, 28, 28)
            torch.nn.Conv2d(in_channels=512, out_channels=1024,
                            kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.Conv2d(in_channels=1024, out_channels=1024,
                            kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.MaxPool2d(kernel_size=2),  # (N, 1024, 14, 14)
            torch.nn.Conv2d(in_channels=1024, out_channels=2048,
                            kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=2048),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.Conv2d(in_channels=2048, out_channels=2048,
                            kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=2048),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.MaxPool2d(kernel_size=2),  # (N, 2048, 7, 7)
            torch.nn.Conv2d(in_channels=2048, out_channels=1024,
                            kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.Conv2d(in_channels=1024, out_channels=512,
                            kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.Conv2d(in_channels=512, out_channels=256,
                            kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.Conv2d(in_channels=256, out_channels=128,
                            kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
        )
        self.flatten = torch.nn.Flatten()
        self.get_enc_mu = torch.nn.Sequential(
            torch.nn.Linear(6272, self.latent_size),
            torch.nn.Tanh()
        )
        self.get_enc_logvar = torch.nn.Sequential(
            torch.nn.Linear(6272, self.latent_size),
            torch.nn.Tanh()
        )

    def forward(self, images, domains, contents):
        """
        Calculates mean and diagonal log-variance of p(z | x).

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
        x = self.flatten(x)
        enc_mu = self.get_enc_mu(x)
        enc_logvar = self.get_enc_logvar(x)
        return enc_mu, enc_logvar


class Decoder(torch.nn.Module):
    def __init__(self, num_domains, num_contents, latent_size):
        super().__init__()
        self.num_domains = num_domains
        self.num_contents = num_contents
        self.latent_size = latent_size
        self.linear = torch.nn.Linear(
            self.latent_size + self.num_domains + self.num_contents, 6272)
        self.dec_conv_sequential = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.ConvTranspose2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.ConvTranspose2d(
                in_channels=512, out_channels=1024, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.ConvTranspose2d(
                in_channels=1024, out_channels=2048, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=2048),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=2048, out_channels=2048,
                                     kernel_size=3, padding=1, bias=False),  # (N, 2048, 14, 14)
            torch.nn.BatchNorm2d(num_features=2048),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.ConvTranspose2d(
                in_channels=2048, out_channels=1024, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=1024, out_channels=1024,
                                     kernel_size=3, padding=1, bias=False),  # (N, 1024, 28, 28)
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                     kernel_size=3, padding=1, bias=False),  # (N, 512, 56, 56)
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                     kernel_size=3, padding=1, bias=False),  # (N, 256, 112, 112)
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
        )
        self.get_dec_mu = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.ConvTranspose2d(
                in_channels=128, out_channels=3, kernel_size=3, padding=1),  # (N, 3, 224, 224)
            torch.nn.Sigmoid(),
        )
        self.get_dec_logvar = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(),
            torch.nn.ConvTranspose2d(
                in_channels=128, out_channels=3, kernel_size=3, padding=1),  # (N, 3, 224, 224)
            torch.nn.Sigmoid(),
        )

    def reshape(self, x):
        return x.view(-1, 128, 7, 7)

    def forward(self, codes, domains, contents):
        """
        Calculates mean and diagonal log-variance of p(x | z).

        codes: Tensor of shape (batch_size, latent_size)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        x = torch.cat((codes, domains, contents), dim=1)
        x = self.linear(x)
        x = self.reshape(x)
        x = self.dec_conv_sequential(x)
        dec_mu = self.get_dec_mu(x)
        dec_logvar = self.get_dec_logvar(x)
        return dec_mu, dec_logvar


if __name__ == "__main__":
    batch_size = 4
    num_domains = 3
    num_contents = 7
    latent_size = 512
    lamb = 10.0
    lr = 0.01
    batch = [
        torch.randn(size=(batch_size, 3, 224, 224)),
        torch.nn.functional.one_hot(torch.randint(low=0, high=num_domains, size=(batch_size,)), num_classes=num_domains),
        torch.nn.functional.one_hot(torch.randint(low=0, high=num_contents, size=(batch_size,)), num_classes=num_contents),
        (f"pic_{i}" for i in range(batch_size))
    ]

    cvae = CVAE(num_domains=num_domains,
                num_contents=num_contents,
                latent_size=latent_size,
                lamb=lamb,
                lr=lr)
    print(cvae)
    train_loss = cvae.training_step(batch, batch_idx=0)
    print(train_loss)
    print("Done!")
