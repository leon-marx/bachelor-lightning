import torch
import pytorch_lightning as pl


def init_glorot_normal(m):
    """
    Glorot normal initialization (AKA xavier normal).
    """
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight, gain=2 ** 0.5)

def init_he_normal(m):
    """
    He normal initialization.
    """
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    
class trVAE(pl.LightningModule):
    def __init__(self, num_domains, num_contents, latent_size, feature_size, mmd_size, dropout_rate, lr, lamb, beta):
        super().__init__()

        self.num_domains = num_domains
        self.num_contents = num_contents
        self.latent_size = latent_size
        self.feature_size = feature_size
        self.mmd_size = mmd_size
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.lamb = lamb
        self.beta = beta
        self.hyper_param_dict = {
            "num_domains": self.num_domains,
            "num_contents": self.num_contents,
            "latent_size": latent_size,
            "feature_size": feature_size,
            "lamb": self.lamb,
            "beta": self.beta,
        }

        self.encoder = Encoder(num_domains=self.num_domains,
                               num_contents=self.num_contents,
                               latent_size=self.latent_size,
                               feature_size=self.feature_size,
                               mmd_size=self.mmd_size,
                               dropout_rate=self.dropout_rate
                               )
        self.decoder = Decoder(num_domains=self.num_domains,
                               num_contents=self.num_contents,
                               latent_size=self.latent_size,
                               feature_size=self.feature_size,
                               mmd_size=self.mmd_size,
                               dropout_rate=self.dropout_rate
                               )

    def loss(self, images, enc_mu, enc_logvar, reconstructions, y_mmd, split_loss=False):
        """
        Calculates the mmd-specific vae loss.

        images: Tensor of shape (batch_size * num_domains, channels, height, width)
        enc_mu: Tensor of shape (batch_size * num_domains, latent_size)
        enc_logvar: Tensor of shape (batch_size * num_domains, latent_size)
        reconstructions: Tensor of shape (batch_size * num_domains, channels, height, width)
        split_loss: bool, if True, returns kld and rec losses separately
        y: Tensor of shape (batch_size * num_domains, mmd_size)
        """
        rec = torch.nn.functional.mse_loss(images, reconstructions, reduction="none").mean(dim=[0, 1, 2, 3])
        kld = self.lamb * 0.5 * (enc_mu ** 2 + enc_logvar.exp() - enc_logvar - 1).mean(dim=[0, 1])
        mmd = 0

        n = int(y_mmd.shape[0] / self.num_domains)
        labeled_y = [y_mmd[i*n:(i+1)*n] for i in range(self.num_domains)]
        sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
        for i in range(len(labeled_y)):
            for j in range(i+1):
                k = 0
                for x1 in labeled_y[i]:
                    for x2 in labeled_y[j]:
                        e = torch.exp(-(x1 - x2) ** 2).mean()
                        for sigma in sigmas:
                            k += e ** sigma * self.beta
                if i == j:
                    mmd += k / n ** 2
                else:
                    mmd -= 2 * k / n ** 2
        if split_loss:
            return kld + rec + mmd, kld.item(), rec.item(), mmd.item()
        else:
            return kld + rec + mmd

    def log(self, name, value, *args, **kwargs):
        print(f"{name}: {value}")

    def forward(self, images, domains, contents):
        """
        Calculates codes for the given images and returns their reconstructions.

        images: Tensor of shape (batch_size * num_domains, channels, height, width)
        domains: Tensor of shape (batch_size * num_domains, num_domains)
        contents: Tensor of shape (batch_size * num_domains, num_contents)
        """
        enc_mu, enc_logvar = self.encoder(images, domains, contents)
        codes = enc_mu + torch.randn_like(enc_mu) * (0.5 * enc_logvar).exp()
        reconstructions, y_mmd = self.decoder(codes, domains, contents)

        return enc_mu, enc_logvar, reconstructions, y_mmd

    def training_step(self, batch, batch_idx):
        """
        Calculates the mmd Loss.

        batch: List [x, domain, content, filenames]
            images: Tensor of shape (batch_size * num_domains, channels, height, width)
            domains: Tensor of shape (batch_size * num_domains, num_domains)
            contents: Tensor of shape (batch_size * num_domains, num_contents)
            filenames: Tuple of strings of the form: {domain}/{content}/{fname}
        batch_idx: The index of the batch, not used.
        """
        images = batch[0]
        domains = batch[1]
        contents = batch[2]

        enc_mu, enc_logvar, reconstructions, y_mmd = self(images, domains, contents)

        loss, kld_value, rec_value, mmd_value = self.loss(
            images, enc_mu, enc_logvar, reconstructions, y_mmd, split_loss=True
        )
        self.log("train_loss", loss, batch_size=images.shape[0])
        self.log("kld", kld_value, prog_bar=True, batch_size=images.shape[0])
        self.log("rec", rec_value, prog_bar=True, batch_size=images.shape[0])
        self.log("mmd", mmd_value, prog_bar=True, batch_size=images.shape[0])
        # self.log("lr", self.optimizers(
        # ).param_groups[0]["lr"], prog_bar=True, batch_size=images.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Calculates the mmd Loss.

        batch: List [x, domain, content, filenames]
            images: Tensor of shape (batch_size * num_domains, channels, height, width)
            domains: Tensor of shape (batch_size * num_domains, num_domains)
            contents: Tensor of shape (batch_size * num_domains, num_contents)
            filenames: Tuple of strings of the form: {domain}/{content}/{fname}
        batch_idx: The index of the batch, not used.
        """
        images = batch[0]
        domains = batch[1]
        contents = batch[2]

        enc_mu, enc_logvar, reconstructions, y_mmd = self(images, domains, contents)

        loss = self.loss(images, enc_mu, enc_logvar, reconstructions, y_mmd)
        self.log("val_loss", loss, batch_size=images.shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        """
        Calculates the mmd Loss.

        batch: List [x, domain, content, filenames]
            images: Tensor of shape (batch_size * num_domains, channels, height, width)
            domains: Tensor of shape (batch_size * num_domains, num_domains)
            contents: Tensor of shape (batch_size * num_domains, num_contents)
            filenames: Tuple of strings of the form: {domain}/{content}/{fname}
        batch_idx: The index of the batch, not used.
        """
        images = batch[0]
        domains = batch[1]
        contents = batch[2]

        enc_mu, enc_logvar, reconstructions, y_mmd = self(images, domains, contents)

        return self.loss(images, enc_mu, enc_logvar, reconstructions, y_mmd)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr)
        return optimizer

    def reconstruct(self, images, domains, contents):
        """
        Calculates codes for the given images and returns their reconstructions.

        images: Tensor of shape (batch_size * num_domains, channels, height, width)
        domains: Tensor of shape (batch_size * num_domains, num_domains)
        contents: Tensor of shape (batch_size * num_domains, num_contents)
        """
        enc_mu, enc_logvar = self.encoder(images, domains, contents)
        reconstructions, y_mmd = self.decoder(enc_mu, domains, contents)

        return reconstructions


class Encoder(torch.nn.Module):
    def __init__(self, num_domains, num_contents, latent_size, feature_size, mmd_size, dropout_rate):
        super().__init__()
        self.num_domains = num_domains
        self.num_contents = num_contents
        self.latent_size = latent_size
        self.feature_size = feature_size
        self.mmd_size = mmd_size
        self.dropout_rate = dropout_rate
        self.encode_domain = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.num_domains, out_features=1024),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1024, out_features=224 * 224),
            torch.nn.ReLU(),
        )
        self.reshape = lambda x: x.view(-1, 1, 224, 224)
        self.enc_conv_sequential = torch.nn.Sequential(
            *self.block(
                in_channels=4,
                out_channels=self.feature_size,
                depth=2,
            ),  # (N, fs, 112, 112)
            *self.block(
                in_channels=self.feature_size,
                out_channels=self.feature_size * 2,
                depth=2,
            ),  # (N, 2fs, 56, 56)
            *self.block(
                in_channels=self.feature_size * 2,
                out_channels=self.feature_size * 4,
                depth=3,
            ),  # (N, 4fs, 28, 28)
            *self.block(
                in_channels=self.feature_size * 4,
                out_channels=self.feature_size * 8,
                depth=3,
            ),  # (N, 8fs, 14, 14)
            *self.block(
                in_channels=self.feature_size * 8,
                out_channels=self.feature_size * 16,
                depth=3,
            ),  # (N, 16fs, 7, 7)
            *self.block(
                in_channels=self.feature_size * 16,
                out_channels=self.feature_size * 32,
                depth=3
            ),  # (N, 32fs, 3, 3)
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=32 * self.feature_size * 9, out_features=2048),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2048, out_features=1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_rate),
        )
        self.get_mu = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=self.latent_size),
        )
        self.get_mu.apply(init_glorot_normal)
        self.get_logvar = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=self.latent_size),
        )
        self.get_logvar.apply(init_glorot_normal)

    def block(self, in_channels, out_channels, depth, last=False):
        seqlist = []
        for i in range(depth):
            if i == 0:
                seq = [
                    torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, padding=1, bias=True),
                    torch.nn.ReLU(),
                ]
            else:
                seq = [
                    torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, padding=1, bias=True),
                    torch.nn.ReLU(),
                ]
            seqlist += seq
        seqlist.append(torch.nn.MaxPool2d(kernel_size=2))
        return seqlist

    def forward(self, images, domains, contents):
        """
        Calculates latent-space encodings for the given images in the form p(z | x).

        images: Tensor of shape (batch_size * num_domains, channels, height, width)
        domains: Tensor of shape (batch_size * num_domains, num_domains)
        contents: Tensor of shape (batch_size * num_domains, num_contents)
        """
        domain_panel = self.encode_domain(domains.float())
        domain_panel = self.reshape(domain_panel)
        x = torch.cat((images, domain_panel), dim=1)
        x = self.enc_conv_sequential(x)
        enc_mu = self.get_mu(x)
        enc_logvar = self.get_logvar(x)

        return enc_mu, enc_logvar


class Decoder(torch.nn.Module):
    def __init__(self, num_domains, num_contents, latent_size, feature_size, mmd_size, dropout_rate):
        super().__init__()
        self.num_domains = num_domains
        self.num_contents = num_contents
        self.latent_size = latent_size
        self.feature_size = feature_size
        self.mmd_size = mmd_size
        self.dropout_rate = dropout_rate
        self.reshape = lambda x: x.view(-1, self.latent_size + self.num_domains, 1, 1)
        self.encode_domain = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.num_domains, out_features=512),
            torch.nn.ReLU(),
        )
        self.get_mmd = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512 + self.latent_size, out_features=self.mmd_size),
            torch.nn.ReLU(),
        )
        self.get_mmd.apply(init_he_normal)
        self.pre_conv = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.mmd_size, out_features=2048),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2048, out_features=32 * self.feature_size * 9),
            torch.nn.ReLU(),
        )
        self.pre_conv.apply(init_he_normal)
        self.reshape = lambda x: x.view(-1, 32 * self.feature_size, 3, 3)
        self.dec_conv_sequential = torch.nn.Sequential(
            *self.block(
                in_channels=self.feature_size * 32,
                out_channels=self.feature_size * 32,
                depth=3
            ),  # (N, 32fs, 7, 7)
            *self.block(
                in_channels=self.feature_size * 32,
                out_channels=self.feature_size * 16,
                depth=3
            ),  # (N, 16fs, 14, 14)
            *self.block(
                in_channels=self.feature_size * 16,
                out_channels=self.feature_size * 8,
                depth=3
            ),  # (N, 8fs, 28, 28)
            *self.block(
                in_channels=self.feature_size * 8,
                out_channels=self.feature_size * 4,
                depth=3
            ),  # (N, 4fs, 56, 56)
            *self.block(
                in_channels=self.feature_size * 4,
                out_channels=self.feature_size * 2,
                depth=2
            ),  # (N, 2fs, 112, 112)
            *self.block(
                in_channels=self.feature_size * 2,
                out_channels=self.feature_size,
                depth=2,
            ),  # (N, fs, 224, 224)
            torch.nn.ConvTranspose2d(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3,
                                        stride=1, padding=1, bias=True),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3,
                                        stride=1, padding=1, bias=True),
            torch.nn.ReLU(),
        )
        self.dec_conv_sequential.apply(init_he_normal)
        self.final_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=self.feature_size, out_channels=3, kernel_size=1,
                                        stride=1, padding=0, bias=True),
            torch.nn.ReLU(),
        )

    def block(self, in_channels, out_channels, depth):
        seqlist = []
        if out_channels == self.feature_size * 32:
            upsample = torch.nn.Upsample(size=7)
        else:
            upsample = torch.nn.Upsample(scale_factor=2)
        for i in range(depth):
            if i == 0:
                seq = [
                    torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, padding=1, bias=True),
                    torch.nn.ReLU(),
                ]
            else:
                seq = [
                    torch.nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, padding=1, bias=True),
                    torch.nn.ReLU(),
                ]
            seqlist += seq
        seqlist.append(upsample)
        return seqlist

    def forward(self, codes, domains, contents):
        """
        Calculates reconstructions of the given latent-space encodings. 

        codes: Tensor of shape (batch_size * num_domains, latent_size)
        domains: Tensor of shape (batch_size * num_domains, num_domains)
        contents: Tensor of shape (batch_size * num_domains, num_contents)
        """
        domain = self.encode_domain(domains.float())
        x = torch.cat((codes, domain), dim=1)
        mmd = self.get_mmd(x)
        preconv = self.pre_conv(mmd)
        preconv = self.reshape(preconv)
        reconstructions = self.dec_conv_sequential(preconv)
        reconstructions = self.final_conv(reconstructions)
        return reconstructions, mmd


if __name__ == "__main__":
    batch_size = 4

    num_domains = 3
    num_contents = 7
    latent_size = 512
    feature_size = 64
    mmd_size = 512
    dropout_rate = 0.2

    lr = 1e-4
    lamb = 1.0
    beta = 1.0

    batch = [
        torch.randn(size=(batch_size * num_domains, 3, 224, 224)),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_domains, size=(batch_size * num_domains,)), num_classes=num_domains),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_contents, size=(batch_size * num_domains,)), num_classes=num_contents),
        (f"pic_{i}" for i in range(batch_size * num_domains))
    ]
    model = trVAE(
        num_domains=num_domains,
        num_contents=num_contents,
        latent_size=latent_size,
        feature_size=feature_size,
        mmd_size=mmd_size,
        dropout_rate=dropout_rate,
        lr=lr,
        lamb=lamb,
        beta=beta)

    # print(model)
    # images, domains, contents, _ = batch
    # domain_panel = model.encoder.encode_domain(domains.float())
    # domain_panel = model.encoder.reshape(domain_panel)
    # output = torch.cat((images, domain_panel), dim=1)
    # for i, m in enumerate(model.encoder.enc_conv_sequential.children()):
    #     output = m(output)
    #     print(m, output.shape)
    # domain = model.decoder.encode_domain(domains.float())
    # x = torch.cat((model.encoder.get_mu(output), domain), dim=1)
    # mmd = model.decoder.get_mmd(x)
    # output = model.decoder.pre_conv(mmd)
    # output = model.decoder.reshape(output)
    # for i, m in enumerate(model.decoder.dec_conv_sequential.children()):
    #     output = m(output)
    #     print(m, output.shape)

    loss = model.training_step(batch, 0)
    print(loss)
    print("Done!")
