from tabnanny import verbose
import torch
import pytorch_lightning as pl


class AE(pl.LightningModule):
    def __init__(self, num_domains, num_contents, latent_size, lr):
        super().__init__()

        self.num_domains = num_domains
        self.num_contents = num_contents
        self.latent_size = latent_size

        self.encoder = Encoder(num_domains=self.num_domains,
                               num_contents=self.num_contents, latent_size=self.latent_size)
        self.decoder = Decoder(num_domains=self.num_domains,
                               num_contents=self.num_contents, latent_size=self.latent_size)

        self.lr = lr

    def loss(self, images, reconstructions):
        """
        Calculates the L1-Norm.

        images: Tensor of shape (batch_size, channels, height, width)
        reconstructions: Tensor of shape (batch_size, channels, height, width)s
        """
        return torch.sum(abs(images - reconstructions))

    def forward(self, images, domains, contents):
        """
        Calculates codes for the given images and returns their reconstructions.

        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        codes = self.encoder(images, domains, contents)
        reconstructions = self.decoder(codes, domains, contents)

        return reconstructions

    def training_step(self, batch, batch_idx):
        """
        Calculates the L1-Norm.

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

        reconstructions = self(images, domains, contents)

        loss = self.loss(images, reconstructions)
        self.log("train_loss", loss, batch_size=images.shape[0])
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True, batch_size=images.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Calculates the L1-Norm.

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

        reconstructions = self(images, domains, contents)

        loss = self.loss(images, reconstructions)
        self.log("val_loss", loss, batch_size=images.shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        """
        Calculates the L1-Norm.

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

        reconstructions = self(images, domains, contents)

        return self.loss(images, reconstructions)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            factor=0.1 ** 0.5,
            patience=2,
            verbose=True
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def reconstruct(self, images, domains, contents):
        """
        Calculates codes for the given images and returns their reconstructions.

        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        codes = self.encoder(images, domains, contents)
        reconstructions = self.decoder(codes, domains, contents)

        return reconstructions


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
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(6272, self.latent_size),
            torch.nn.Tanh()
        )

    def forward(self, images, domains, contents):
        """
        Calculates latent-space encodings for the given images.

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
        codes = self.linear(x)

        return codes


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
        Calculates reconstructions of the given latent-space encodings. 

        codes: Tensor of shape (batch_size, latent_size)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        x = torch.cat((codes, domains, contents), dim=1)
        x = self.linear(x)
        x = self.reshape(x)
        reconstructions = self.dec_conv_sequential(x)
        return reconstructions


if __name__ == "__main__":
    batch_size = 4
    num_domains = 3
    num_contents = 7
    latent_size = 512
    lr = 0.01
    batch = [
        torch.randn(size=(batch_size, 3, 224, 224)),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_domains, size=(batch_size,)), num_classes=num_domains),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_contents, size=(batch_size,)), num_classes=num_contents),
        (f"pic_{i}" for i in range(batch_size))
    ]

    model = AE(num_domains=num_domains,
               num_contents=num_contents,
               latent_size=latent_size,
               lr=lr)
    print(model)
    train_loss = model.training_step(batch, batch_idx=0)
    print(train_loss)
    print("Done!")
