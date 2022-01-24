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



class CVAE_v4(pl.LightningModule):
    def __init__(self, num_domains, num_contents, lr, depth, out_channels, kernel_size, activation, downsampling, upsampling, dropout, batch_norm, loss_mode, lamb, level, no_bn_last=True):
        super().__init__()

        self.num_domains = num_domains
        self.num_contents = num_contents
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
        self.level = level
        self.no_bn_last = no_bn_last
        self.hyper_param_dict = {
            "num_domains": self.num_domains,
            "num_contents": self.num_contents,
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
            "level": self.level,
            "no_bn_last": self.no_bn_last,
        }

        self.encoder = Encoder(num_domains=self.num_domains,
                               num_contents=self.num_contents,
                               depth=self.depth,
                               out_channels=self.out_channels,
                               kernel_size=self.kernel_size,
                               activation=self.activation,
                               downsampling=self.downsampling,
                               dropout=self.dropout,
                               batch_norm=self.batch_norm,
                               level=self.level,
        )
        self.decoder = Decoder(num_domains=self.num_domains,
                               num_contents=self.num_contents,
                               depth=self.depth,
                               out_channels=self.out_channels[::-1],
                               kernel_size=self.kernel_size,
                               activation=self.activation,
                               upsampling=self.upsampling,
                               dropout=self.dropout,
                               batch_norm=self.batch_norm,
                               no_bn_last=self.no_bn_last,
                               level=self.level,
        )

        self.lr = lr
        if isinstance(activation, torch.nn.SELU):
            self.apply(selu_init)

    def loss(self, images, enc_mu, enc_logvar, reconstructions):
        """
        Calculates the loss. Choose from l1, l2 and elbo

        images: Tensor of shape (batch_size, channels, height, width)
        Calculates the ELBO Loss (negative ELBO).
        
        images: Tensor of shape (batch_size, channels, height, width)
        enc_mu: Tensor of shape (batch_size, out_channels[6], 4, 4)
        enc_logvar: Tensor of shape (batch_size, out_channels[6], 4, 4)
        reconstructions: Tensor of shape (batch_size, channels, height, width)
        """
        if self.loss_mode == "l1":
            loss = torch.abs(images - reconstructions)
            return loss.mean(dim=[0, 1, 2, 3])
        if self.loss_mode == "l2":
            loss = torch.nn.functional.mse_loss(
                images, reconstructions, reduction="none")
            return loss.mean(dim=[0, 1, 2, 3])
        if self.loss_mode == "elbo":
            kld = self.lamb * 0.5 * (enc_mu ** 2 + enc_logvar.exp() - enc_logvar - 1).mean(dim=[0, 1, 2, 3])
            rec = torch.nn.functional.mse_loss(images, reconstructions, reduction="none").mean(dim=[0, 1, 2, 3])
            return kld + rec

    def set_level(self, level):
        self.level = level
        self.encoder.level = level
        self.decoder.level = level
        for param in self.parameters():
            param.requires_grad = False
        if self.level < 8:
            for param in self.encoder.enc_conv_blocks[self.level-1].parameters():
                param.requires_grad = True
            for param in self.decoder.dec_conv_blocks[-self.level].parameters():
                param.requires_grad = True
        if self.level == 8:
            for param in self.encoder.get_mu.parameters():
                param.requires_grad = True
            for param in self.encoder.get_logvar.parameters():
                param.requires_grad = True
            for param in self.decoder.dec_conv_blocks[-self.level].parameters():
                param.requires_grad = True

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

        print(f"images: {images.shape}")
        print(f"enc_mu: {enc_mu.shape}")
        print(f"enc_logvar: {enc_logvar.shape}")
        print(f"codes: {codes.shape}")
        print(f"reconstructions: {reconstructions.shape}")

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

        loss = self.loss(images, enc_mu, enc_logvar, reconstructions)
        self.log("train_loss", loss, batch_size=images.shape[0])
        self.log("lr", self.optimizers(
        ).param_groups[0]["lr"], prog_bar=True, batch_size=images.shape[0])
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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


class Encoder(torch.nn.Module):
    def __init__(self, num_domains, num_contents, depth, out_channels, kernel_size, activation, downsampling, dropout, batch_norm, level):
        super().__init__()
        self.num_domains = num_domains
        self.num_contents = num_contents
        self.depth = depth
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.downsampling = downsampling
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.level = level
        self.size_dict = {
            1: 224,
            2: 224,
            3: 112,
            4: 56,
            5: 28,
            6: 14,
            7: 7,
            8: 4,
        }
        self.enc_conv_blocks = torch.nn.ModuleList([
            self.block(
                depth=self.depth,
                in_channels=3 + self.num_domains + self.num_contents,
                out_channels=self.out_channels[0],
                kernel_size=self.kernel_size,
                activation=self.activation,
                downsampling="none",
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [0], 224, 224)
            self.block(
                depth=self.depth,
                in_channels=self.out_channels[0] + self.num_domains + self.num_contents,
                out_channels=self.out_channels[1],
                kernel_size=self.kernel_size,
                activation=self.activation,
                downsampling=self.downsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [1], 112, 112)
            self.block(
                depth=self.depth,
                in_channels=self.out_channels[1] + self.num_domains + self.num_contents,
                out_channels=self.out_channels[2],
                kernel_size=self.kernel_size,
                activation=self.activation,
                downsampling=self.downsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [2], 56, 56)
            self.block(
                depth=self.depth,
                in_channels=self.out_channels[2] + self.num_domains + self.num_contents,
                out_channels=self.out_channels[3],
                kernel_size=self.kernel_size,
                activation=self.activation,
                downsampling=self.downsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [3], 28, 28)
            self.block(
                depth=self.depth,
                in_channels=self.out_channels[3] + self.num_domains + self.num_contents,
                out_channels=self.out_channels[4],
                kernel_size=self.kernel_size,
                activation=self.activation,
                downsampling=self.downsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [4], 14, 14)
            self.block(
                depth=self.depth,
                in_channels=self.out_channels[4] + self.num_domains + self.num_contents,
                out_channels=self.out_channels[5],
                kernel_size=self.kernel_size,
                activation=self.activation,
                downsampling=self.downsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [5], 7, 7)
            self.block(
                depth=self.depth,
                in_channels=self.out_channels[5] + self.num_domains + self.num_contents,
                out_channels=self.out_channels[6],
                kernel_size=self.kernel_size,
                activation=self.activation,
                downsampling=self.downsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [6], 4, 4))
        ])
        self.get_mu = self.block(
                depth=self.depth,
                in_channels=self.out_channels[6] + self.num_domains + self.num_contents,
                out_channels=self.out_channels[7],
                kernel_size=self.kernel_size,
                activation=self.activation,
                downsampling="none",
                dropout=self.dropout,
                batch_norm=self.batch_norm
        )  # (N, [6], 4, 4))
        self.get_logvar = self.block(
                depth=self.depth,
                in_channels=self.out_channels[6] + self.num_domains + self.num_contents,
                out_channels=self.out_channels[7],
                kernel_size=self.kernel_size,
                activation=self.activation,
                downsampling="none",
                dropout=self.dropout,
                batch_norm=self.batch_norm
        )  # (N, [6], 4, 4))

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
        return torch.nn.Sequential(*seq_list)

    def forward(self, images, domains, contents):
        """
        Calculates latent-space encodings for the given images in the form p(z | x).

        images: Tensor of shape (batch_size, channels, height, width)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        x = images
        for i, seq in enumerate(self.enc_conv_blocks[:self.level]):
            domain_panels = torch.ones(size=(x.shape[0], self.num_domains, self.size_dict[i+1], self.size_dict[i+1])).to(
                x.device) * domains.view(x.shape[0], self.num_domains, 1, 1)
            content_panels = torch.ones(size=(x.shape[0], self.num_contents, self.size_dict[i+1], self.size_dict[i+1])).to(
                x.device) * contents.view(x.shape[0], self.num_contents, 1, 1)
            x = seq(torch.cat((x, domain_panels, content_panels), dim=1))
        if self.level >= 8:
            domain_panels = torch.ones(size=(x.shape[0], self.num_domains, self.size_dict[8], self.size_dict[8])).to(
                x.device) * domains.view(x.shape[0], self.num_domains, 1, 1)
            content_panels = torch.ones(size=(x.shape[0], self.num_contents, self.size_dict[8], self.size_dict[8])).to(
                x.device) * contents.view(x.shape[0], self.num_contents, 1, 1)
            enc_mu = self.get_mu(torch.cat((x, domain_panels, content_panels), dim=1))
            enc_logvar = self.get_logvar(torch.cat((x, domain_panels, content_panels), dim=1))
        else:
            enc_mu = x
            enc_logvar = torch.zeros_like(x)

        return enc_mu, enc_logvar


class Decoder(torch.nn.Module):
    def __init__(self, num_domains, num_contents, depth, out_channels, kernel_size, activation, upsampling, dropout, batch_norm, no_bn_last, level):
        super().__init__()
        self.num_domains = num_domains
        self.num_contents = num_contents
        self.depth = depth
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.upsampling = upsampling
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.no_bn_last = no_bn_last
        self.level = level
        self.size_dict = {
            1: 224,
            2: 112,
            3: 56,
            4: 28,
            5: 14,
            6: 7,
            7: 4,
            8: 4,
        }
        self.dec_conv_blocks = torch.nn.ModuleList([
            self.block(
                depth=self.depth,
                in_channels=self.out_channels[0] + self.num_domains + self.num_contents,
                out_channels=self.out_channels[1],
                kernel_size=self.kernel_size,
                activation=self.activation,
                upsampling="none",
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [1], 4, 4)
            self.block(
                depth=self.depth,
                in_channels=self.out_channels[1] + self.num_domains + self.num_contents,
                out_channels=self.out_channels[2],
                kernel_size=self.kernel_size,
                activation=self.activation,
                upsampling=self.upsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm,
                output_padding=0
            ),  # (N, [2], 7, 7)
            self.block(
                depth=self.depth,
                in_channels=self.out_channels[2] + self.num_domains + self.num_contents,
                out_channels=self.out_channels[3],
                kernel_size=self.kernel_size,
                activation=self.activation,
                upsampling=self.upsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [3], 14, 14)
            self.block(
                depth=self.depth,
                in_channels=self.out_channels[3] + self.num_domains + self.num_contents,
                out_channels=self.out_channels[4],
                kernel_size=self.kernel_size,
                activation=self.activation,
                upsampling=self.upsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [4], 28, 28)
            self.block(
                depth=self.depth,
                in_channels=self.out_channels[4] + self.num_domains + self.num_contents,
                out_channels=self.out_channels[5],
                kernel_size=self.kernel_size,
                activation=self.activation,
                upsampling=self.upsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            ),  # (N, [5], 56, 56)
            self.block(
                depth=self.depth,
                in_channels=self.out_channels[5] + self.num_domains + self.num_contents,
                out_channels=self.out_channels[6],
                kernel_size=self.kernel_size,
                activation=self.activation,
                upsampling=self.upsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm,
                last_block=self.no_bn_last
            ),  # (N, [6], 112, 112)
            self.block(
                depth=self.depth,
                in_channels=self.out_channels[6] + self.num_domains + self.num_contents,
                out_channels=self.out_channels[7],
                kernel_size=self.kernel_size,
                activation=self.activation,
                upsampling=self.upsampling,
                dropout=self.dropout,
                batch_norm=self.batch_norm,
                last_block=self.no_bn_last
            ),  # (N, [7], 224, 224)
            self.block(
                depth=self.depth,
                in_channels=self.out_channels[7] + self.num_domains + self.num_contents,
                out_channels=3,
                kernel_size=self.kernel_size,
                activation=self.activation,
                upsampling="none",
                dropout=self.dropout,
                batch_norm=self.batch_norm,
                last_block=self.no_bn_last
            ),  # (N, 3, 224, 224)
        ])

    def block(self, depth, in_channels, out_channels, kernel_size, activation, upsampling="stride", dropout=False, batch_norm=False, last_block=False, output_padding=1):
        seq_list = []
        if isinstance(activation, torch.nn.SELU):
            dropout = False
            batch_norm = False
        for i in range(depth):
            seq = []
            if i == 0: # upsampling in first layer of block
                if upsampling == "stride":
                    seq.append(torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        padding=int((kernel_size-1)/2), output_padding=output_padding, stride=2, bias=not batch_norm))
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
        return torch.nn.Sequential(*seq_list)

    def forward(self, codes, domains, contents):
        """
        Calculates reconstructions of the given latent-space encodings. 

        codes: Tensor of shape (batch_size, out_channels[6], 4, 4)
        domains: Tensor of shape (batch_size, num_domains)
        contents: Tensor of shape (batch_size, num_contents)
        """
        x = codes
        for i, seq in enumerate(self.dec_conv_blocks[-self.level:]):
            domain_panels = torch.ones(size=(x.shape[0], self.num_domains, self.size_dict[self.level-i], self.size_dict[self.level-i])).to(
                x.device) * domains.view(x.shape[0], self.num_domains, 1, 1)
            content_panels = torch.ones(size=(x.shape[0], self.num_contents, self.size_dict[self.level-i], self.size_dict[self.level-i])).to(
                x.device) * contents.view(x.shape[0], self.num_contents, 1, 1)
            x = seq(torch.cat((x, domain_panels, content_panels), dim=1))

        return x


if __name__ == "__main__":
    batch_size = 4
    num_domains = 3
    num_contents = 7
    
    lr = 1e-4
    out_channels = [256, 256, 512, 512, 1024, 1024, 2048, 2048]

    depth = 3
    kernel_size = 3
    activation = torch.nn.SELU()
    downsampling = "stride"
    upsampling = "stride"
    dropout = False
    batch_norm = False
    loss_mode = "elbo"
    lamb = 0

    batch = [
        torch.randn(size=(batch_size, 3, 224, 224)),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_domains, size=(batch_size,)), num_classes=num_domains),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_contents, size=(batch_size,)), num_classes=num_contents),
        (f"pic_{i}" for i in range(batch_size))
    ]
    model = CVAE_v4(num_domains=num_domains, num_contents=num_contents, lr=lr, depth=depth, 
        out_channels=out_channels, kernel_size=kernel_size, activation=activation,
        downsampling=downsampling, upsampling=upsampling, dropout=dropout,
        batch_norm=batch_norm, loss_mode=loss_mode, lamb=lamb, level=1)
    for level in range(7, 9, 1):
        print(f"level: {level}")
        model.set_level(level)
        model(batch[0], batch[1], batch[2])
        print("parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("    ", name)
            else:
                print("        ", name)
        print("")
        print("")
