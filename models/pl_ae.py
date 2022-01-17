import torch
import pytorch_lightning as pl


class PL_AE(pl.LightningModule):
    def __init__(self, num_domains, num_contents, latent_size, lr):
        super().__init__()

        self.num_domains = num_domains
        self.num_contents = num_contents
        self.latent_size = latent_size

        self.encoder = Encoder(num_domains=self.num_domains, num_contents=self.num_contents, latent_size=self.latent_size)
        self.decoder = Decoder(num_domains=self.num_domains, num_contents=self.num_contents, latent_size=self.latent_size)

        self.lr = lr

    def loss(self, images, reconstructions):
        """
        Calculates the MSE Loss.

        images: Tensor of shape (batch_size, channels, height, width)
        reconstructions: Tensor of shape (batch_size, channels, height, width)s
        """
        loss = torch.nn.functional.mse_loss(images, reconstructions, reduction="none")
        return loss.sum(dim=[1, 2, 3]).mean(dim=[0])
    
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
        Calculates the MSE Loss.

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
        Calculates the MSE Loss.

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
        Calculates the MSE Loss.

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
            factor=0.1,
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
                            out_channels=128, kernel_size=3, padding=1, stride=2),  # 224x224 => 112x112
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=128,
                            out_channels=128, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=128,
                            out_channels=256, kernel_size=3, padding=1, stride=2),  # 112x112 => 56x56
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=256, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=512, kernel_size=3, padding=1, stride=2),  # 56x56 => 28x28
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=512,
                            out_channels=512, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=512,
                            out_channels=512, kernel_size=3, padding=1, stride=2),  # 28x28 => 14x14
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=512,
                            out_channels=512, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=512,
                            out_channels=512, kernel_size=3, padding=1, stride=2),  # 14x14 => 7x7
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=512,
                            out_channels=256, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=128, kernel_size=3, padding=1),
            torch.nn.GELU(),
        )
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(49 * 128, self.latent_size)

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
        self.linear = torch.nn.Linear(self.latent_size + self.num_domains + self.num_contents, 49 * 128)
        self.dec_conv_sequential = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=128,
                            out_channels=256, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=512, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, output_padding=1, padding=1, stride=2),  # 7x7 => 14x14
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, output_padding=1, padding=1, stride=2),  # 14x14 => 28x28
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, output_padding=1, padding=1, stride=2),  # 28x28 => 56x56
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, output_padding=1, padding=1, stride=2),  # 56x56 => 112x112
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, output_padding=1, padding=1, stride=2),  # 112x112 => 224x224
            torch.nn.Sigmoid(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
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
        torch.nn.functional.one_hot(torch.randint(low=0, high=num_domains, size=(batch_size,)), num_classes=num_domains),
        torch.nn.functional.one_hot(torch.randint(low=0, high=num_contents, size=(batch_size,)), num_classes=num_contents),
        (f"pic_{i}" for i in range(batch_size))
    ]

    model = PL_AE(num_domains=num_domains,
                num_contents=num_contents,
                latent_size=latent_size,
                lr=lr)
    print(model)
    train_loss = model.training_step(batch, batch_idx=0)
    print(train_loss)
    print("Done!")
