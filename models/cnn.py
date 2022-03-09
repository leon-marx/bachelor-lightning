import torch
import pytorch_lightning as pl


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


class CNN(pl.LightningModule):
    def __init__(self, data, num_domains, num_contents, latent_size, lr, depth, out_channels, kernel_size, activation, downsampling, dropout, batch_norm, initialize=False):
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
        self.dropout = dropout
        self.batch_norm = batch_norm
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
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
        }

        self.featurizer = Encoder(data=self.data,
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
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.latent_size, 1024),
            self.activation,
            torch.nn.Linear(1024, 128),
            self.activation,
            torch.nn.Linear(128, self.num_contents),
            self.activation,
        )

        self.lr = lr
        if initialize:
            if isinstance(activation, torch.nn.SELU):
                self.apply(selu_init)

    def forward(self, images):
        """
        Calculates class predictions for a given batch of images.

        images: Tensor of shape (batch_size, channels, height, width)
        """
        features = self.featurizer(images)
        predictions = self.classifier(features)
        return predictions

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
        contents = batch[2]

        loss = torch.nn.functional.cross_entropy(self(images), torch.argmax(contents, dim=1))
        # self.log("train_loss", loss)
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
        contents = batch[2]

        loss = torch.nn.functional.cross_entropy(self(images), torch.argmax(contents, dim=1))
        # self.log("val_loss", loss)
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
        contents = batch[2]

        loss = torch.nn.functional.cross_entropy(self(images), torch.argmax(contents, dim=1))
        # self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


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
                    in_channels=self.C,
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
            self.get_features = torch.nn.Linear(49 * self.out_channels[5], self.latent_size)
        if self.data == "RMNIST":
            self.enc_conv_sequential = torch.nn.Sequential(
                *self.block(
                    depth=self.depth,
                    in_channels=self.C,
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
            )
            self.flatten = torch.nn.Flatten()
            self.get_features = torch.nn.Linear(49 * self.out_channels[2], self.latent_size)

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

    def forward(self, images):
        """
        Extracts features of a given batch of images.

        images: Tensor of shape (batch_size, channels, height, width)
        """
        x = self.enc_conv_sequential(images)
        x = self.flatten(x)
        features = self.get_features(x)

        return features


if __name__ == "__main__":
    batch_size = 1
    num_domains = 6
    num_contents = 10

    lr = 1e-4
    out_channels = [128, 256, 512, 512, 1024, 1024]

    latent_size = 128
    depth = 1
    kernel_size = 3
    activation = torch.nn.ELU()
    downsampling = "stride"
    dropout = False
    batch_norm = True

    batch = [
        torch.randn(size=(batch_size, 1, 28, 28)),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_domains, size=(batch_size,)), num_classes=num_domains),
        torch.nn.functional.one_hot(torch.randint(
            low=0, high=num_contents, size=(batch_size,)), num_classes=num_contents),
        (f"pic_{i}" for i in range(batch_size))
    ]

    model = CNN(data="RMNIST", num_domains=num_domains, num_contents=num_contents,
        latent_size=latent_size, lr=lr, depth=depth,
        out_channels=out_channels, kernel_size=kernel_size, activation=activation,
        downsampling=downsampling, dropout=dropout,
        batch_norm=batch_norm)
    # ae_loss = model.training_step(batch, 0)
    # print(ae_loss)
    # print("Done!")


    # from datasets.rotated_mnist import RMNISTDataModule
    root = "data"#/variants/RMNIST_augmented"
    domains = [0, 15, 30, 45, 75]
    contents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # dm = RMNISTDataModule(root=root, domains=domains, contents=contents,
    #                     batch_size=batch_size, num_workers=0)
    # dm.setup()
    # batch = next(iter(dm.train_dataloader()))
    # loss = model.training_step(batch, 0)
    # print(loss)
    # print("Done!")
    output = model(batch[0])
    pred = torch.argmax(output).item()
    print(output.shape)
    print(pred)
