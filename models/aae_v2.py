from matplotlib import image
import torch
import pytorch_lightning as pl
import lpips


def selu_init(m):
    """
    LeCun Normal initialization for selu.
    """
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class AAE_v2(pl.LightningModule):
    def __init__(self, data, num_domains, num_contents, latent_size, lr, depth, out_channels, kernel_size, activation, downsampling, upsampling, dropout, batch_norm, loss_mode, lamb, net, calibration, no_bn_last=True, initialize=False):
        super().__init__()
        self.data = data
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
        self.no_bn_last = no_bn_last
        self.lamb = lamb
        self.net = net
        self.calibration = calibration
        self.get_mse_loss = torch.nn.MSELoss(reduction="mean")
        self.get_bce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.hyper_param_dict = {
            "data": self.data,
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
            "no_bn_last": self.no_bn_last,
        }

        self.encoder = Encoder(
            data=self.data,
            num_domains=self.num_domains,
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
        self.decoder = Decoder(
            data=self.data,
            num_domains=self.num_domains,
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
        self.discriminator = Discriminator(
            latent_size=self.latent_size,
            activation = self.activation,
            dropout = self.dropout
            )

        self.lr = lr
        if initialize:
            if isinstance(activation, torch.nn.SELU):
                self.apply(selu_init)
        if self.loss_mode == "deep_lpips":
            self.lpips = lpips.LPIPS(net=self.net, lpips=self.calibration)

    def vae_loss(self, images, reconstructions, codes, codes_2=None, split_loss=False):
        """
        Calculates the l2 loss..

        images: Tensor of shape (batch_size, channels, height, width)
        reconstructions: Tensor of shape (batch_size, channels, height, width)
        split_loss: bool, if True, returns kld and rec losses separately
        """
        if "deep" in self.loss_mode:
            if self.loss_mode == "deep_own":
                img_loss = self.get_mse_loss(images, reconstructions)
                code_loss = self.get_mse_loss(codes, codes_2)
                self.log("deep_loss_img", img_loss.item(), batch_size=images.shape[0], logger=True)
                self.log("deep_loss_code", code_loss.item(), batch_size=images.shape[0], logger=True)
                loss = img_loss + code_loss

            elif self.loss_mode == "deep_lpips":
                lpips_loss = self.lamb * self.lpips(images, reconstructions).mean()
                image_loss = (1 - self.lamb) * self.get_mse_loss(images, reconstructions)
                loss = lpips_loss + image_loss

            if split_loss:
                return loss, loss.item()
            else:
                return loss
        else:
            loss = self.get_mse_loss(images, reconstructions)
            if split_loss:
                return loss, loss.item()
            else:
                return loss

    def disc_loss(self, pred, truth, split_loss=False):
        """
        Calculates the discriminator loss.

        pred: Tensor of shape (batch_size)
        truth: Tensor of shape (batch_size)
        split_loss: bool, if True, also returns the value
        """
        loss = self.get_bce_loss(pred, truth)
        if split_loss:
            return loss, loss.item()
        else:
            return loss

    def forward(self, images, domains, contents):
        """
        Calculates codes for the given images and returns their reconstructions.

        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        codes = self.encoder(images, domains, contents)
        reconstructions = self.decoder(codes, domains, contents)

        return codes, reconstructions

    def training_step(self, batch, batch_idx, optimizer_idx):
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

        codes, reconstructions = self(images, domains, contents)

        # Train CVAE for reconstruction
        if optimizer_idx == 0:
            if self.loss_mode == "deep_own":
                with torch.no_grad():
                    codes_2 = self.encoder(reconstructions, domains, contents)
                loss , value = self.vae_loss(images, reconstructions, codes, codes_2=codes_2, split_loss=True)
            elif self.loss_mode == "deep_lpips":
                loss , value = self.vae_loss(images, reconstructions, codes, split_loss=True)
            else:
                loss , value = self.vae_loss(images, reconstructions, codes, split_loss=True)
            self.log("rec_train_loss", loss, batch_size=images.shape[0])
            self.log("rec", value, batch_size=images.shape[0], prog_bar=True)
            return loss

        # Train Discriminator for regularization
        if optimizer_idx == 1:
            real_latent_noise = torch.randn_like(codes).to(self.device)

            real_pred = self.discriminator(real_latent_noise)
            real_truth = torch.ones_like(real_pred).to(self.device) * 0.9
            real_loss, real_value = self.disc_loss(real_pred, real_truth, split_loss=True)

            fake_pred = self.discriminator(codes.detach())
            fake_truth = torch.ones_like(fake_pred).to(self.device) * 0.1
            fake_loss, fake_value = self.disc_loss(fake_pred, fake_truth, split_loss=True)


            loss = real_loss + fake_loss
            self.log("disc_train_loss", loss, batch_size=images.shape[0])
            self.log("dis", real_value + fake_value, batch_size=images.shape[0], prog_bar=True)
            return loss

        # Train Encoder for confusion
        if optimizer_idx == 2:

            confusion_pred = self.discriminator(codes)
            confusion_truth = torch.ones_like(confusion_pred).to(self.device) * 0.9
            loss, value = self.disc_loss(confusion_pred, confusion_truth, split_loss=True)

            self.log("enc_train_loss", loss, batch_size=images.shape[0])
            self.log("con", value, batch_size=images.shape[0], prog_bar=True)
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

        codes, reconstructions = self(images, domains, contents)

        if self.loss_mode == "deep_own":
            codes_2 = self.encoder(reconstructions, domains, contents)
            loss = self.vae_loss(images, reconstructions, codes, codes_2=codes_2)

        elif self.loss_mode == "deep_lpips":
            loss = self.vae_loss(images, reconstructions, codes)
        else:
            loss = self.vae_loss(images, reconstructions, codes)
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

        codes, reconstructions = self(images, domains, contents)

        return self.vae_loss(images, reconstructions, codes)

    def warmer(self):
        if self.lamb < 1.0:
            self.lamb *= 10 ** 0.5
            print(f"New lambda: {self.lamb}")

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(params=list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_e = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [opt_ae, opt_d, opt_e], []

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

    def transfer(self, images, domains, contents, decoder_domains, decoder_contents):
        """
        Calculates codes for the given images and returns their reconstructions.

        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        decoder_domains: Tensor of shape (batch_size, num_domains)
        decoder_contents: Tensor of shape (batch_size, num_contents)
        """
        codes = self.encoder(images, domains, contents)
        transfers = self.decoder(codes, decoder_domains, decoder_contents)

        return transfers

    def generate(self, codes, domains, contents):
        """
        Generate images from Gaussian distributed codes.
        """
        with torch.no_grad():
            self.eval()
            reconstructions = self.decoder(codes, domains, contents)
            self.train()
            return reconstructions

class Encoder(torch.nn.Module):
    def __init__(self, data, num_domains, num_contents, latent_size, depth, out_channels, kernel_size, activation, downsampling, dropout, batch_norm):
        super().__init__()
        self.data = data
        self.HW = {"PACS": 224, "RMNIST": 28}[self.data]
        self.C = {"PACS": 3, "RMNIST": 1}[self.data]
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
        if self.data == "PACS":
            self.enc_conv_sequential = torch.nn.Sequential(
                *self.block(
                    depth=self.depth,
                    in_channels=self.C + self.num_domains + self.num_contents,
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
                *self.block(
                    depth=self.depth,
                    in_channels=self.out_channels[5],
                    out_channels=self.out_channels[6],
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    downsampling=self.downsampling,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                ),  # (N, [6], 4, 4)
            )
            self.flatten = torch.nn.Flatten()
            self.get_code = torch.nn.Sequential(
                torch.nn.Linear(16 * self.out_channels[6], self.latent_size),
                self.activation,
            )
        if self.data == "RMNIST":
            self.enc_conv_sequential = torch.nn.Sequential(
                *self.block(
                    depth=self.depth,
                    in_channels=self.C + self.num_domains + self.num_contents,
                    out_channels=self.out_channels[0],
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    downsampling="none",
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                ),  # (N, [0], 28, 28)
                *self.block(
                    depth=self.depth,
                    in_channels=self.out_channels[0],
                    out_channels=self.out_channels[1],
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    downsampling=self.downsampling,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                ),  # (N, [1], 14, 14)
                *self.block(
                    depth=self.depth,
                    in_channels=self.out_channels[1],
                    out_channels=self.out_channels[2],
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    downsampling=self.downsampling,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                ),  # (N, [2], 7, 7)
                *self.block(
                    depth=self.depth,
                    in_channels=self.out_channels[2],
                    out_channels=self.out_channels[3],
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    downsampling=self.downsampling,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                ),  # (N, [3], 4, 4)
            )
            self.flatten = torch.nn.Flatten()
            self.get_code = torch.nn.Sequential(
                torch.nn.Linear(16 * self.out_channels[3], self.latent_size),
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
        domain_panels = torch.ones(size=(images.shape[0], self.num_domains, self.HW, self.HW)).to(
            images.device) * domains.view(images.shape[0], self.num_domains, 1, 1)
        content_panels = torch.ones(size=(images.shape[0], self.num_contents, self.HW, self.HW)).to(
            images.device) * contents.view(images.shape[0], self.num_contents, 1, 1)

        x = torch.cat((images, domain_panels, content_panels), dim=1)
        x = self.enc_conv_sequential(x)
        x = self.flatten(x)
        codes= self.get_code(x)

        return codes

class Decoder(torch.nn.Module):
    def __init__(self, data, num_domains, num_contents, latent_size, depth, out_channels, kernel_size, activation, upsampling, dropout, batch_norm, no_bn_last):
        super().__init__()
        self.data = data
        self.HW = {"PACS": 224, "RMNIST": 28}[self.data]
        self.C = {"PACS": 3, "RMNIST": 1}[self.data]
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
            torch.nn.Linear(self.latent_size + self.num_domains + self.num_contents, 16 * self.out_channels[0]),
            self.activation,
        )
        self.reshape = lambda x: x.view(-1, self.out_channels[0], 4, 4)
        if self.data == "PACS":
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
                ),  # (N, [1], 4, 4)
                *self.block(
                    depth=self.depth,
                    in_channels=self.out_channels[1],
                    out_channels=self.out_channels[2],
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    upsampling=self.upsampling,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
                    special_seven=True
                ),  # (N, [2], 7, 7)
                *self.block(
                    depth=self.depth,
                    in_channels=self.out_channels[2],
                    out_channels=self.out_channels[3],
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    upsampling=self.upsampling,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                ),  # (N, [3], 14, 14)
                *self.block(
                    depth=self.depth,
                    in_channels=self.out_channels[3],
                    out_channels=self.out_channels[4],
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    upsampling=self.upsampling,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                ),  # (N, [4], 28, 28)
                *self.block(
                    depth=self.depth,
                    in_channels=self.out_channels[4],
                    out_channels=self.out_channels[5],
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    upsampling=self.upsampling,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                ),  # (N, [5], 56, 56)
                *self.block(
                    depth=self.depth,
                    in_channels=self.out_channels[5],
                    out_channels=self.out_channels[6],
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    upsampling=self.upsampling,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                ),  # (N, [6], 112, 112)
                *self.block(
                    depth=self.depth,
                    in_channels=self.out_channels[6],
                    out_channels=self.C,
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    upsampling=self.upsampling,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
                    last_block=self.no_bn_last
                ),  # (N, self.C, 224, 224)
            )
        if self.data == "RMNIST":
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
                ),  # (N, [1], 4, 4)
                *self.block(
                    depth=self.depth,
                    in_channels=self.out_channels[1],
                    out_channels=self.out_channels[2],
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    upsampling=self.upsampling,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
                    special_seven=True
                ),  # (N, [2], 7, 7)
                *self.block(
                    depth=self.depth,
                    in_channels=self.out_channels[2],
                    out_channels=self.out_channels[3],
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    upsampling=self.upsampling,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                ),  # (N, [3], 14, 14)
                *self.block(
                    depth=self.depth,
                    in_channels=self.out_channels[3],
                    out_channels=self.C,
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    upsampling=self.upsampling,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
                    last_block=self.no_bn_last
                ),  # (N, self.C, 28, 28)
            )

    def block(self, depth, in_channels, out_channels, kernel_size, activation, upsampling="stride", dropout=False, batch_norm=False, last_block=False, special_seven=False):
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
                    if special_seven:
                        seq.append(torch.nn.Upsample(size=7, mode="nearest"))
                    else:
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
        x = self.linear(x)
        x = self.reshape(x)
        reconstructions = self.dec_conv_sequential(x)
        return reconstructions

class Discriminator(torch.nn.Module):
    def __init__(self, latent_size, activation, dropout):
        super().__init__()
        self.latent_size = latent_size
        self.activation = activation
        self.dropout = dropout
        if self.dropout:
            self.sequential = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.latent_size, out_features=1024),
                self.activation,
                torch.nn.Dropout(),
                torch.nn.Linear(in_features=1024, out_features=2048),
                self.activation,
                torch.nn.Dropout(),
                torch.nn.Linear(in_features=2048, out_features=1024),
                self.activation,
                torch.nn.Dropout(),
                torch.nn.Linear(in_features=1024, out_features=1),
            )
        else:
            self.sequential = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.latent_size, out_features=1024),
                self.activation,
                torch.nn.Linear(in_features=1024, out_features=2048),
                self.activation,
                torch.nn.Linear(in_features=2048, out_features=1024),
                self.activation,
                torch.nn.Linear(in_features=1024, out_features=1),
            )
    def forward(self, codes):
        """
        codes: Tensor of shape (2 * batch_size, latent_size)
        """
        logits = self.sequential(codes)
        return logits


if __name__ == "__main__":
    batch_size = 4
    num_domains = 3
    num_contents = 7

    lr = 1e-4
    # out_channels = [128, 256, 512, 512, 1024, 1024, 2048]
    out_channels = [512, 512, 512, 512, 512, 512, 512]
#

    latent_size = 128
    depth = 1
    kernel_size = 3
    activation = torch.nn.ELU()
    downsampling = "maxpool"
    upsampling = "stride"
    dropout = False
    batch_norm = True
    loss_mode = "deep_lpips"
    lamb = 0.1
    net = "alex"
    calibration = True

    batch = [
        torch.randn(size=(batch_size, 3, 224, 224)),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_domains, size=(batch_size,)), num_classes=num_domains),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_contents, size=(batch_size,)), num_classes=num_contents),
        (f"pic_{i}" for i in range(batch_size))
    ]
    model = AAE_v2(data="PACS", num_domains=num_domains, num_contents=num_contents,
        latent_size=latent_size, lr=lr, depth=depth,
        out_channels=out_channels, kernel_size=kernel_size, activation=activation,
        downsampling=downsampling, upsampling=upsampling, dropout=dropout, loss_mode=loss_mode,
        lamb=lamb, net=net, calibration=calibration, batch_norm=batch_norm)


    # Analyzing the model layers and outputs
    images = batch[0]
    domains = batch[1]
    contents = batch[2]
    domain_panels = torch.ones(size=(images.shape[0], num_domains, 224, 224)).to(
        images.device) * domains.view(images.shape[0], num_domains, 1, 1)
    content_panels = torch.ones(size=(images.shape[0], num_contents, 224, 224)).to(
        images.device) * contents.view(images.shape[0], num_contents, 1, 1)

    output = torch.cat((images, domain_panels, content_panels), dim=1)
    for i, m in enumerate(model.encoder.enc_conv_sequential.children()):
        output = m(output)
        print(m, output.shape)
    codes = model.encoder.get_code(model.encoder.flatten(output))
    output = torch.cat((codes, domains, contents), dim=1)
    output = model.decoder.reshape(model.decoder.linear(output))
    for i, m in enumerate(model.decoder.dec_conv_sequential.children()):
        output = m(output)
        print(m, output.shape)


    ae_loss = model.training_step(batch, 0, 0)
    print(f"ae_loss: {ae_loss}")
    disc_loss = model.training_step(batch, 0, 1)
    print(f"disc_loss: {disc_loss}")
    print("Done!")
