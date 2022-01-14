from xml.dom.pulldom import PROCESSING_INSTRUCTION
import torch
import pytorch_lightning as pl


class CVAE(pl.LightningModule):
    def __init__(self, num_classes, num_domains, latent_size, lamb, lr):
        super().__init__()

        self.num_classes = num_classes
        self.num_domains = num_domains
        self.latent_size = latent_size

        self.encoder = Encoder(num_classes=self.num_classes,
                               num_domains=self.num_domains, latent_size=self.latent_size)
        self.decoder = Decoder(num_classes=self.num_classes,
                               num_domains=self.num_domains, latent_size=self.latent_size)

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

    def forward(self, images, classes, domains, raw=False):
        """
        Calculates mean and diagonal log-variance of p(z | x) and of p(x | z).

        images: Tensor of shape (batch_size, channels, height, width)
        classes: Tensor of shape (batch_size, num_classes)
        domains: Tensor of shape (batch_size, num_domains)
        raw: Bool, if True: z is sampled without noise
        """
        enc_mu, enc_logvar = self.encoder(images, classes, domains)

        if raw:
            codes = enc_mu
        else:
            z_std = (0.5 * enc_logvar).exp()
            z_eps = torch.randn_like(enc_mu)
            codes = enc_mu + z_eps * z_std

        dec_mu, dec_logvar = self.decoder(codes, classes, domains)

        return enc_mu, enc_logvar, dec_mu, dec_logvar

    def training_step(self, batch, batch_idx):
        """
        Calculates the ELBO Loss (negative ELBO).

        batch: List of tuples [(x, y)]
            x: {"image": Tensor of shape (batch_size, channels, height, width),
                "domain": Tensor of shape (batch_size)}
                    The values correspond to int d = 0,...,2 (domain)
            y: Tensor of shape (batch_size)
                The values correspond to int d = 0,...,6 (class)
        batch_idx: The index of the batch, not used.
        """
        images = torch.cat([x["image"] for x, y in batch]
                           )  # (batch_size, channels, height, width)
        classes = torch.nn.functional.one_hot(torch.cat(
            [y for x, y in batch]), num_classes=self.num_classes).flatten(start_dim=1)  # (batch_size, num_classes)
        domains = torch.nn.functional.one_hot(torch.cat(
            [x["domain"] for x, y in batch]), num_classes=self.num_domains).flatten(start_dim=1)  # (batch_size, num_domains)

        enc_mu, enc_logvar, dec_mu, dec_logvar = self(images, classes, domains)

        return self.loss(images, enc_mu, enc_logvar, dec_mu, dec_logvar)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class Encoder(torch.nn.Module):
    def __init__(self, num_classes, num_domains, latent_size):
        super().__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.latent_size = latent_size
        self.enc_conv_sequential = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3 + self.num_classes + self.num_domains,
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

    def forward(self, images, classes, domains):
        """
        Calculates mean and diagonal log-variance of p(z | x).

        images: Tensor of shape (batch_size, channels, height, width)
        classes: Tensor of shape (batch_size, num_classes)
        domains: Tensor of shape (batch_size, num_domains)
        """
        class_panels = torch.ones(size=(images.shape[0], self.num_classes, 224, 224)).to(
            images.device) * classes.view(images.shape[0], self.num_classes, 1, 1)
        domain_panels = torch.ones(size=(images.shape[0], self.num_domains, 224, 224)).to(
            images.device) * domains.view(images.shape[0], self.num_domains, 1, 1)
        x = torch.cat((images, class_panels, domain_panels), dim=1)
        x = self.enc_conv_sequential(x)
        x = self.flatten(x)
        enc_mu = self.get_enc_mu(x)
        enc_logvar = self.get_enc_logvar(x)
        return enc_mu, enc_logvar


class Decoder(torch.nn.Module):
    def __init__(self, num_classes, num_domains, latent_size):
        super().__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.latent_size = latent_size
        self.linear = torch.nn.Linear(
            self.latent_size + self.num_classes + self.num_domains, 6272)
        self.reshape = lambda x: x.view(-1, 128, 7, 7)
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

    def forward(self, codes, classes, domains):
        """
        Calculates mean and diagonal log-variance of p(x | z).

        codes: Tensor of shape (batch_size, latent_size)
        classes: Tensor of shape (batch_size, num_classes)
        domains: Tensor of shape (batch_size, num_domains)
        """
        x = torch.cat((codes, classes, domains), dim=1)
        x = self.linear(x)
        x = self.reshape(x)
        x = self.dec_conv_sequential(x)
        dec_mu = self.get_dec_mu(x)
        dec_logvar = self.get_dec_logvar(x)
        return dec_mu, dec_logvar


if __name__ == "__main__":
    batch_size = 4
    batch = []
    for i in range(3):  # iterating over all datasets / environments
        x = {"image": torch.randn(size=(batch_size, 3, 224, 224)),
             "domain": torch.randint(low=0, high=2, size=(batch_size,))}
        y = torch.randint(low=0, high=7, size=(batch_size,))
        batch.append((x, y))

    num_classes = 7
    num_domains = 3
    latent_size = 512
    lamb = 10.0
    lr = 0.01
    cvae = CVAE(num_classes=num_classes,
                num_domains=num_domains,
                latent_size=latent_size,
                lamb=lamb,
                lr=lr)
    print(cvae)
    train_loss = cvae.training_step(batch, batch_idx=0)
    print(train_loss)
    print("Done!")
