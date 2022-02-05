import torch
import pytorch_lightning as pl


def selu_init(m):
    """
    LeCun Normal initialization for selu.
    """
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        torch.nn.init.zeros_(m.bias)


class MMD_CVAE(pl.LightningModule):
    def __init__(self, num_domains, num_contents, latent_size, lr, depth, out_channels, kernel_size, activation, downsampling, upsampling, dropout, batch_norm, loss_mode, lamb, beta, no_bn_last=True):
        super().__init__()

        self.num_domains = num_domains
        self.num_contents = num_contents
        self.latent_size = latent_size
        self.depth = depth
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.downsampling = downsampling
        self.upsampling = upsampling
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.loss_mode = loss_mode
        self.lamb = lamb
        self.beta = beta
        self.no_bn_last = no_bn_last
        self.hyper_param_dict = {
            "num_domains": self.num_domains,
            "num_contents": self.num_contents,
            "latent_size": self.latent_size,
            "depth": self.depth,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "downsampling": self.downsampling,
            "upsampling": self.upsampling,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "loss_mode": self.loss_mode,
            "lamb": self.lamb,
            "beta": self.beta,
            "no_bn_last": self.no_bn_last,
        }

        self.encoder = Encoder(num_domains=self.num_domains,
                               num_contents=self.num_contents,
                               latent_size=self.latent_size,
                               depth=self.depth,
                               out_channels=self.out_channels,
                               kernel_size=self.kernel_size,
                               activation=self.activation,
                               downsampling=self.downsampling,
                               dropout=self.dropout,
                               batch_norm=self.batch_norm
        )
        self.decoder = Decoder(num_domains=self.num_domains,
                               num_contents=self.num_contents,
                               latent_size=self.latent_size,
                               depth=self.depth,
                               out_channels=self.out_channels[::-1],
                               kernel_size=self.kernel_size,
                               activation=self.activation,
                               upsampling=self.upsampling,
                               dropout=self.dropout,
                               batch_norm=self.batch_norm,
                               no_bn_last=self.no_bn_last
        )

        self.lr = lr
        if isinstance(activation, torch.nn.SELU):
            self.apply(selu_init)

    def loss(self, images, enc_mu, enc_logvar, reconstructions, y_mmd, codes_2=None, split_loss=False):
        """
        Calculates the loss. Choose from l1, l2 and elbo

        images: Tensor of shape (batch_size, channels, height, width)
        enc_mu: Tensor of shape (batch_size, latent_size)
        enc_logvar: Tensor of shape (batch_size, latent_size)
        reconstructions: Tensor of shape (batch_size, channels, height, width)
        split_loss: bool, if True, returns kld and rec losses separately
        y_mmd: Tensor of shape (batch_size * num_domains, mmd_size)
        """
        if self.loss_mode == "mmd":
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
        if self.loss_mode == "deep":
            img_loss = self.get_mse_loss(images, reconstructions)
            code_mu_loss = self.get_mse_loss(enc_mu, codes_2[0])
            code_logvar_loss = self.get_mse_loss(enc_mu, codes_2[1])
            rec = img_loss + code_mu_loss + code_logvar_loss
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
        reconstructions, y_mmd = self.decoder(codes, domains, contents)

        return enc_mu, enc_logvar, reconstructions, y_mmd

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

        enc_mu, enc_logvar, reconstructions, y_mmd = self(images, domains, contents)
        if self.loss_mode == "deep":
            codes_2 = self.encoder(reconstructions, domains, contents)
            loss, kld_value, rec_value, mmd_value = self.loss(images, enc_mu, enc_logvar, reconstructions, y_mmd, codes_2=codes_2, split_loss=True)
        else:
            loss, kld_value, rec_value, mmd_value = self.loss(images, enc_mu, enc_logvar, reconstructions, y_mmd, split_loss=True)

        self.log("train_loss", loss, batch_size=images.shape[0])
        self.log("kld", kld_value, prog_bar=True, batch_size=images.shape[0])
        self.log("rec", rec_value, prog_bar=True, batch_size=images.shape[0])
        self.log("mmd", mmd_value, prog_bar=True, batch_size=images.shape[0])
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

        enc_mu, enc_logvar, reconstructions, y_mmd = self(images, domains, contents)

        if self.loss_mode == "deep":
            codes_2 = self.encoder(reconstructions, domains, contents)
            loss = self.loss(images, enc_mu, enc_logvar, reconstructions, y_mmd, codes_2=codes_2)
        else:
            loss = self.loss(images, enc_mu, enc_logvar, reconstructions, y_mmd)
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

        enc_mu, enc_logvar, reconstructions, y_mmd = self(images, domains, contents)

        return self.loss(images, enc_mu, enc_logvar, reconstructions, y_mmd)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def warmer(self):
        self.lamb *= 10 ** 0.5
        print(f"New lambda: {self.lamb}")
        self.beta *= 10 ** 0.5
        print(f"New beta: {self.beta}")

    def reconstruct(self, images, domains, contents):
        """
        Calculates codes for the given images and returns their reconstructions.

        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        enc_mu, enc_logvar = self.encoder(images, domains, contents)
        reconstructions, y_mmd = self.decoder(enc_mu, domains, contents)

        return reconstructions

    def transfer(self, images, domains, contents, decoder_domains):
        """
        Calculates codes for the given images and returns their reconstructions.

        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        decoder_domains: Tensor of shape (batch_size, num_domains)
        """
        enc_mu, enc_logvar = self.encoder(images, domains, contents)
        reconstructions, y_mmd = self.decoder(enc_mu, decoder_domains, contents)

        return reconstructions

    def generate(self, codes, domains, contents):
        """
        Generate images from Gaussian distributed codes.
        """
        with torch.no_grad():
            self.eval()
            x = torch.cat((codes, domains, contents), dim=1)
            x = self.decoder.linear(x)
            x = self.decoder.reshape(x)
            reconstructions = self.decoder.dec_conv_sequential(x)
            self.train()
            return reconstructions

class Encoder(torch.nn.Module):
    def __init__(self, num_domains, num_contents, latent_size, depth, out_channels, kernel_size, activation, downsampling, dropout, batch_norm):
        super().__init__()
        self.num_domains = num_domains
        self.num_contents = num_contents
        self.latent_size = latent_size
        self.depth = depth
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.downsampling = downsampling
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.enc_conv_sequential = torch.nn.Sequential(
            *self.block(
                depth=self.depth,
                in_channels=3 + self.num_domains + self.num_contents,
                out_channels=self.out_channels[0],
                kernel_size=self.kernel_size,
                activation=self.activation,
                downsampling="none",
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [0], 224, 224)
            *self.block(
                depth=self.depth,
                in_channels=self.out_channels[0],
                out_channels=self.out_channels[1],
                kernel_size=self.kernel_size,
                activation=self.activation,
                downsampling=self.downsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [1], 112, 112)
            *self.block(
                depth=self.depth,
                in_channels=self.out_channels[1],
                out_channels=self.out_channels[2],
                kernel_size=self.kernel_size,
                activation=self.activation,
                downsampling=self.downsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [2], 56, 56)
            *self.block(
                depth=self.depth,
                in_channels=self.out_channels[2],
                out_channels=self.out_channels[3],
                kernel_size=self.kernel_size,
                activation=self.activation,
                downsampling=self.downsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [3], 28, 28)
            *self.block(
                depth=self.depth,
                in_channels=self.out_channels[3],
                out_channels=self.out_channels[4],
                kernel_size=self.kernel_size,
                activation=self.activation,
                downsampling=self.downsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [4], 14, 14)
            *self.block(
                depth=self.depth,
                in_channels=self.out_channels[4],
                out_channels=self.out_channels[5],
                kernel_size=self.kernel_size,
                activation=self.activation,
                downsampling=self.downsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [5], 7, 7)
        )
        self.flatten = torch.nn.Flatten()
        self.get_mu = torch.nn.Sequential(
            torch.nn.Linear(49 * self.out_channels[5], self.latent_size),
            self.activation,
        )
        self.get_logvar = torch.nn.Sequential(
            torch.nn.Linear(49 * self.out_channels[5], self.latent_size),
            self.activation,
        )

    def block(self, depth, in_channels, out_channels, kernel_size, activation, downsampling="stride", dropout=False, batch_norm=False):
        seq_list = []
        if isinstance(activation, torch.nn.SELU):
            dropout = False
            batch_norm = False
        for i in range(depth):
            seq = []
            if i == 0: # downsampling in first layer of block
                if downsampling == "stride":
                    seq.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        padding=int((kernel_size-1)/2), stride=2, bias=not batch_norm))
                    if batch_norm:
                        seq.append(torch.nn.BatchNorm2d(num_features=out_channels))
                    seq.append(activation)
                    if dropout:
                        seq.append(torch.nn.Dropout2d())
                    seq_list += seq
                elif downsampling == "maxpool":
                    seq.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        padding=int((kernel_size-1)/2), stride=1, bias=not batch_norm))
                    if batch_norm:
                        seq.append(torch.nn.BatchNorm2d(num_features=out_channels))
                    seq.append(activation)
                    seq.append(torch.nn.MaxPool2d(kernel_size=2))
                    if dropout:
                        seq.append(torch.nn.Dropout2d())
                    seq_list += seq
                elif downsampling == "none":
                    seq.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        padding=int((kernel_size-1)/2), stride=1, bias=not batch_norm))
                    if batch_norm:
                        seq.append(torch.nn.BatchNorm2d(num_features=out_channels))
                    seq.append(activation)
                    if dropout:
                        seq.append(torch.nn.Dropout2d())
                    seq_list += seq
            else:
                seq.append(torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    padding=int((kernel_size-1)/2), stride=1, bias=not batch_norm))
                if batch_norm:
                    seq.append(torch.nn.BatchNorm2d(num_features=out_channels))
                seq.append(activation)
                if dropout:
                    seq.append(torch.nn.Dropout2d())
                seq_list += seq       
        return seq_list

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
        x = self.flatten(x)
        enc_mu = self.get_mu(x)
        enc_logvar = self.get_logvar(x)

        return enc_mu, enc_logvar

class Decoder(torch.nn.Module):
    def __init__(self, num_domains, num_contents, latent_size, depth, out_channels, kernel_size, activation, upsampling, dropout, batch_norm, no_bn_last):
        super().__init__()
        self.num_domains = num_domains
        self.num_contents = num_contents
        self.latent_size = latent_size
        self.depth = depth
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.upsampling = upsampling
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.no_bn_last = no_bn_last
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(self.latent_size + self.num_domains + self.num_contents, 49 * self.out_channels[0]),
            self.activation,
        )
        self.reshape = lambda x: x.view(-1, self.out_channels[0], 7, 7)
        self.dec_conv_sequential = torch.nn.Sequential(
            *self.block(
                depth=self.depth,
                in_channels=self.out_channels[0],
                out_channels=self.out_channels[1],
                kernel_size=self.kernel_size,
                activation=self.activation,
                upsampling="none",
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [1], 7, 7)
            *self.block(
                depth=self.depth,
                in_channels=self.out_channels[1],
                out_channels=self.out_channels[2],
                kernel_size=self.kernel_size,
                activation=self.activation,
                upsampling=self.upsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [2], 14, 14)
            *self.block(
                depth=self.depth,
                in_channels=self.out_channels[2],
                out_channels=self.out_channels[3],
                kernel_size=self.kernel_size,
                activation=self.activation,
                upsampling=self.upsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [3], 28, 28)
            *self.block(
                depth=self.depth,
                in_channels=self.out_channels[3],
                out_channels=self.out_channels[4],
                kernel_size=self.kernel_size,
                activation=self.activation,
                upsampling=self.upsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [4], 56, 56)
            *self.block(
                depth=self.depth,
                in_channels=self.out_channels[4],
                out_channels=self.out_channels[5],
                kernel_size=self.kernel_size,
                activation=self.activation,
                upsampling=self.upsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [5], 112, 112)
            *self.block(
                depth=self.depth,
                in_channels=self.out_channels[5],
                out_channels=3,
                kernel_size=self.kernel_size,
                activation=self.activation,
                upsampling=self.upsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm,
                last_block=self.no_bn_last
            ),  # (N, 3, 224, 224)
        )

    def block(self, depth, in_channels, out_channels, kernel_size, activation, upsampling="stride", dropout=False, batch_norm=False, last_block=False):
        seq_list = []
        if isinstance(activation, torch.nn.SELU):
            dropout = False
            batch_norm = False
        for i in range(depth):
            seq = []
            if i == 0: # upsampling in first layer of block
                if upsampling == "stride":
                    seq.append(torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4,
                                        padding=int((kernel_size-1)/2), output_padding=1, stride=2, bias=not batch_norm))
                    if batch_norm:
                        if not (i == depth - 1 and last_block):
                            seq.append(torch.nn.BatchNorm2d(num_features=out_channels))
                    seq.append(activation)
                    if dropout:
                        if not (i == depth - 1 and last_block):
                            seq.append(torch.nn.Dropout2d())
                    seq_list += seq
                elif upsampling == "upsample":
                    seq.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        padding=int((kernel_size-1)/2), stride=1, bias=not batch_norm))
                    if batch_norm:
                        if not (i == depth - 1 and last_block):
                            seq.append(torch.nn.BatchNorm2d(num_features=out_channels))
                    seq.append(activation)
                    seq.append(torch.nn.Upsample(scale_factor=2, mode="nearest"))
                    if dropout:
                        if not (i == depth - 1 and last_block):
                            seq.append(torch.nn.Dropout2d())
                    seq_list += seq
                elif upsampling == "none":
                    seq.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        padding=int((kernel_size-1)/2), stride=1, bias=not batch_norm))
                    if batch_norm:
                        if not (i == depth - 1 and last_block):
                            seq.append(torch.nn.BatchNorm2d(num_features=out_channels))
                    seq.append(activation)
                    if dropout:
                        if not (i == depth - 1 and last_block):
                            seq.append(torch.nn.Dropout2d())
                    seq_list += seq
            else:
                seq.append(torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    padding=int((kernel_size-1)/2), stride=1, bias=not batch_norm))
                if batch_norm:
                    if not (i == depth - 1 and last_block):
                        seq.append(torch.nn.BatchNorm2d(num_features=out_channels))
                seq.append(activation)
                if dropout:
                    if not (i == depth - 1 and last_block):
                        seq.append(torch.nn.Dropout2d())
                seq_list += seq       
        return seq_list

    def forward(self, codes, domains, contents):
        """
        Calculates reconstructions of the given latent-space encodings. 

        codes: Tensor of shape (batch_size, latent_size)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        x = torch.cat((codes, domains, contents), dim=1)
        y_mmd = self.linear(x)
        x = self.reshape(y_mmd)
        reconstructions = self.dec_conv_sequential(x)
        return reconstructions, y_mmd


if __name__ == "__main__":
    batch_size = 4
    num_domains = 3
    num_contents = 7
    
    lr = 1e-4
    out_channels = [128, 256, 512, 512, 1024, 1024]

    latent_size = 128
    depth = 1
    kernel_size = 3
    activation = torch.nn.ELU()
    downsampling = "stride"
    upsampling = "upsample"
    dropout = False
    batch_norm = True
    loss_mode = "mmd"
    lamb = 1.0
    beta = 1.0

    batch = [
        torch.randn(size=(batch_size, 3, 224, 224)),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_domains, size=(batch_size,)), num_classes=num_domains),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_contents, size=(batch_size,)), num_classes=num_contents),
        (f"pic_{i}" for i in range(batch_size))
    ]
    model = MMD_CVAE(
        num_domains=num_domains, num_contents=num_contents,
        latent_size=latent_size, lr=lr, depth=depth, 
        out_channels=out_channels, kernel_size=kernel_size, activation=activation,
        downsampling=downsampling, upsampling=upsampling, dropout=dropout,
        batch_norm=batch_norm, loss_mode=loss_mode, lamb=lamb, beta=beta
        )
    print("Done!")
