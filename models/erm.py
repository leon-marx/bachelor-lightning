import torch
import pytorch_lightning as pl


class MNIST_CNN(torch.nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = torch.nn.GroupNorm(8, 64)
        self.bn1 = torch.nn.GroupNorm(8, 128)
        self.bn2 = torch.nn.GroupNorm(8, 128)
        self.bn3 = torch.nn.GroupNorm(8, 128)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x

def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)

class ERM(pl.LightningModule):
    def __init__(self, input_shape, num_classes, nonlinear_classifier, lr, weight_decay):
        super().__init__()
        self.featurizer = MNIST_CNN(input_shape)
        self.classifier = Classifier(self.featurizer.n_outputs, num_classes, nonlinear_classifier)

        self.network = torch.nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.hyper_param_dict = {
            "num_classes": num_classes,
            "weight_decay": weight_decay,
            "nonlinear_classifier": nonlinear_classifier,
        }

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        images = batch[0]
        domains = batch[1]
        contents = batch[2]

        loss = torch.nn.functional.cross_entropy(self(images), torch.argmax(contents, dim=1))
        # self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch[0]
        domains = batch[1]
        contents = batch[2]

        loss = torch.nn.functional.cross_entropy(self(images), torch.argmax(contents, dim=1))
        # self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        images = batch[0]
        contents = batch[2]

        loss = torch.nn.functional.cross_entropy(self(images), torch.argmax(contents, dim=1))
        # self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer


if __name__ == "__main__":
    batch_size = 4
    num_domains = 6
    num_contents = 10

    input_shape = (1, 28, 28)
    num_classes = num_contents
    nonlinear_classifier = False
    lr = 1e-4
    weight_decay = 0
    hparams=None

    # batch = [
    #     torch.randn(size=(batch_size, 1, 28, 28)),
    #     torch.nn.functional.one_hot(torch.randint(
    #         low=0, high=num_domains, size=(batch_size,)), num_classes=num_domains),
    #     torch.nn.functional.one_hot(torch.randint(
    #         low=0, high=num_contents, size=(batch_size,)), num_classes=num_contents),
    #     (f"pic_{i}" for i in range(batch_size))
    # ]

    model = ERM(
        input_shape=input_shape,
        num_classes=num_classes,
        nonlinear_classifier=nonlinear_classifier,
        lr=lr,
        weight_decay=weight_decay)
    # loss = model.training_step(batch, 0)
    # print(loss)
    # print("Done!")


    from datasets.rotated_mnist import RMNISTDataModule
    root = "data/variants/RMNIST_augmented"
    domains = [0, 15, 30, 45, 75]
    contents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dm = RMNISTDataModule(root=root, domains=domains, contents=contents,
                        batch_size=batch_size, num_workers=0)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    loss = model.training_step(batch, 0)
    print(loss)
    print("Done!")
