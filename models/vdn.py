import torch
import pytorch_lightning as pl

class DVN(pl.LightningModule):
    def __init__(self, latent_size, lambdas, loss_mode, lr, adv_freq):
        super().__init__()
        self.hyper_param_dict = {}

    def forward(self, images, domains, contents):
        """
        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        images = batch[0]
        domains = batch[1]
        contents = batch[2]

        # Encodings
        z_c = self.task_encoder(images)
        z_d = self.domain_encoder(images)
        z_d_hat = self.domain_encoder(images[torch.randperm(images.shape[0])])

        # Reconstructions
        x_rec = self.generator(z_c, z_d)
        x_rand = self.generator(z_c, z_d_hat)

        # Task Loss
        rec_classification = self.classifier(x_rec)
        rand_classification = self.classifier(x_rand)
        l_task = 0.0
        l_task += torch.nn.functional.cross_entropy(rec_classification, torch.argmax(contents, dim=1))
        l_task += torch.nn.functional.cross_entropy(rand_classification, torch.argmax(contents, dim=1))

        # Regularization Loss
        z_disc_true = self.code_discriminator(z_c)
        z_disc_fake = self.code_discriminator()
        l_reg = 0

        # Gan Loss
        x_disc_true = self.image_discriminator(images)
        x_disc_fake = self.image_discriminator(x_rec)
        l_gan = 0
        l_gan += torch.nn.functional.binary_cross_entropy(x_disc_true, torch.ones_like(x_disc_true).to(x_disc_true.device), reduction="mean")  # try torch.ones * 0.9
        l_gan += torch.nn.functional.binary_cross_entropy(x_disc_fake, torch.zeros_like(x_disc_fake).to(x_disc_fake.device), reduction="mean")  # try torch.ones * 0.1

        # Reconstruction Loss
        l_rec = 0
        if self.loss_mode == "l1":
            l_rec += torch.abs(images - x_rec).mean()
        elif self.loss_mode == "perceptual":
            l_rec += torch.abs(self.perceptor(images) - self.perceptor(x_rec)).mean()

        # Total
        l_gen = l_task + self.lambdas["reg"] * l_reg + self.lambdas["gan"] * l_gan + self.lambdas["rec"] * l_rec




    def training_step(self, batch, batch_idx):
        """
        batch: List [x, domain, content, filenames]
            images: Tensor of shape (batch_size, channels, height, width)
            domains: Tensor of shape (batch_size, num_domains)
            contents: Tensor of shape (batch_size, num_contents)
            filenames: Tuple of strings of the form: {domain}/{content}/{fname}
        batch_idx: The index of the batch, not used.
        """
        return loss

    def validation_step(self, batch, batch_idx):
        """
        batch: List [x, domain, content, filenames]
            images: Tensor of shape (batch_size, channels, height, width)
            domains: Tensor of shape (batch_size, num_domains)
            contents: Tensor of shape (batch_size, num_contents)
            filenames: Tuple of strings of the form: {domain}/{content}/{fname}
        batch_idx: The index of the batch, not used.
        """
        with torch.no_grad():
            self.eval()

            self.train()
            return loss

    def test_step(self, batch, batch_idx):
        """
        batch: List [x, domain, content, filenames]
            images: Tensor of shape (batch_size, channels, height, width)
            domains: Tensor of shape (batch_size, num_domains)
            contents: Tensor of shape (batch_size, num_contents)
            filenames: Tuple of strings of the form: {domain}/{content}/{fname}
        batch_idx: The index of the batch, not used.
        """
        with torch.no_grad():
            self.eval()

            self.train()
            return loss

    def configure_optimizers(self):
        pass

    def reconstruct(self, images, domains, contents):
        """
        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        with torch.no_grad():
            self.eval()

            self.train()
            return reconstructions

    def transfer(self, images, domains, contents, decoder_domains, decoder_contents):
        """
        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        decoder_domains: Tensor of shape (batch_size, num_domains)
        decoder_contents: Tensor of shape (batch_size, num_contents)
        """
        with torch.no_grad():
            self.eval()

            self.train()
            return reconstructions

    def generate(self, codes, domains, contents):
        """

        """
        with torch.no_grad():
            self.eval()

            self.train()
            return reconstructions

class DomainEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images, domains, contents):
        """
        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        pass

class TaskEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images, domains, contents):
        """
        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        pass

class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images, domains, contents):
        """
        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        pass

class ImageDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images, domains, contents):
        """
        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        pass

class CodeDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images, domains, contents):
        """
        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        pass

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images, domains, contents):
        """
        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        pass

class Perceptor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images, domains, contents):
        """
        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        pass


if __name__ == "__main__":
    batch_size = 4
    num_domains = 3
    num_contents = 7

    batch = [
        torch.randn(size=(batch_size, 3, 224, 224)),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_domains, size=(batch_size,)), num_classes=num_domains),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_contents, size=(batch_size,)), num_classes=num_contents),
        (f"pic_{i}" for i in range(batch_size))
    ]
    model = DVN()

    loss = model.training_step(batch, 0)
    print(loss)
    print("Done!")
