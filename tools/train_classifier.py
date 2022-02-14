from argparse import ArgumentParser
import os
import pytorch_lightning as pl
import torch

# Own Modules
from datasets.pacs import PACSDataModule
from datasets.rotated_mnist import RMNISTDataModule
from models.erm import ERM
from models.cnn import CNN
from callbacks.classification_logger import ClassificationLogger


if __name__ == "__main__":
    # Parser
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--domains", type=str, default=None)
    # Model
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-04)
    parser.add_argument("--ckpt_path", type=str, default="0")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--out_channels", type=str, default="128,256,512,512,1024,1024")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--downsampling", type=str, default="stride")
    parser.add_argument("--dropout", action="store_true", default=False)
    parser.add_argument("--batch_norm", action="store_true", default=False)
    parser.add_argument("--nonlinear_classifier", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # Training
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--disable_checkpointing", action="store_true", default=False)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=17)
    
    args = parser.parse_args()

    # Configuration
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    pl.seed_everything(args.random_seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Printing Configuration
    print("Environment:")
    print(f"    PyTorch: {torch.__version__}")
    print(f"    CUDA: {torch.version.cuda}")
    print(f"    CUDNN: {torch.backends.cudnn.version()}")
    print("Args:")
    for k, v in sorted(vars(args).items()):
        print(f"    {k}: {v}")
    
    # Dataset
    batch_size = args.batch_size
    if args.domains is None:
        argument_domains = ["a"]
    else:
        argument_domains = [char for char in args.domains]
    if args.data == "PACS":
        domain_dict = {
            "a": ["art_painting", "cartoon", "photo"],
            "0": ["art_painting"],
            "1": ["cartoon"],
            "2": ["photo"],
            "3": ["sketch"],
        }
        domains = []
        for key in argument_domains:
            domains += domain_dict[key]
        domains = sorted(domains)
        contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        dm = PACSDataModule(root="data", domains=domains, contents=contents,
                            batch_size=batch_size, num_workers=args.num_workers)
        log_dm = PACSDataModule(root="data", domains=domains, contents=contents,
                            batch_size=batch_size, num_workers=args.num_workers, shuffle_all=True)
    elif args.data == "PACS_small":
        domain_dict = {
            "a": ["art_painting", "cartoon", "photo"],
            "0": ["art_painting"],
            "1": ["cartoon"],
            "2": ["photo"],
            "3": ["sketch"],
        }
        domains = []
        for key in argument_domains:
            domains += domain_dict[key]
        domains = sorted(domains)
        contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        dm = PACSDataModule(root="data/variants/PACS_small", domains=domains, contents=contents,
                            batch_size=batch_size, num_workers=args.num_workers)
        log_dm = PACSDataModule(root="data/variants/PACS_small", domains=domains, contents=contents,
                            batch_size=batch_size, num_workers=args.num_workers, shuffle_all=True)
    elif args.data == "RMNIST":
        domain_dict = {
            "a": [0, 15, 30, 45, 60, 75],
            "0": [0],
            "1": [15],
            "2": [30],
            "3": [45],
            "4": [60],
            "5": [75],
        }
        domains = []
        for key in argument_domains:
            domains += domain_dict[key]
        domains = sorted(domains)
        contents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        dm = RMNISTDataModule(root="data", domains=domains, contents=contents,
                            batch_size=batch_size, num_workers=args.num_workers)
        log_dm = RMNISTDataModule(root="data", domains=domains, contents=contents,
                            batch_size=batch_size, num_workers=args.num_workers, shuffle_all=True)
    elif args.data == "RMNIST_augmented":
        domain_dict = {
            "a": [0, 15, 30, 45, 60, 75],
            "0": [0],
            "1": [15],
            "2": [30],
            "3": [45],
            "4": [60],
            "5": [75],
        }
        domains = []
        for key in argument_domains:
            domains += domain_dict[key]
        domains = sorted(domains)
        contents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        dm = RMNISTDataModule(root="data/variants/RMNIST_augmented", domains=domains, contents=contents,
                            batch_size=batch_size, num_workers=args.num_workers)
        log_dm = RMNISTDataModule(root="data/variants/RMNIST_augmented", domains=domains, contents=contents,
                            batch_size=batch_size, num_workers=args.num_workers, shuffle_all=True)
    log_dm.setup()
    train_batch = next(iter(log_dm.train_dataloader()))
    val_batch = next(iter(log_dm.val_dataloader()))

    # Model
    num_domains = len(domains)
    num_contents = len(contents)
    latent_size = args.latent_size
    lr = args.lr
    depth = args.depth
    out_channels = list(map(int, args.out_channels.split(",")))
    kernel_size = args.kernel_size
    activation = {
        "relu": torch.nn.ReLU(),
        "gelu": torch.nn.GELU(),
        "lrelu": torch.nn.LeakyReLU(),
        "elu": torch.nn.ELU(),
        "selu": torch.nn.SELU(),
    }[args.activation]
    downsampling = args.downsampling
    dropout = args.dropout
    batch_norm = args.batch_norm
    data = args.data
    nonlinear_classifier = args.nonlinear_classifier
    weight_decay = args.weight_decay
    if args.ckpt_path != "0":
        if args.model == "CNN":
            model = CNN.load_from_checkpoint(args.ckpt_path, 
                data="RMNIST",
                num_domains=num_domains,
                num_contents=num_contents,
                latent_size=latent_size,
                lr=lr,
                depth=depth, 
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                downsampling=downsampling,
                dropout=dropout,
                batch_norm=batch_norm)
        if args.model == "ERM":
            model = ERM.load_from_checkpoint(args.ckpt_path, 
                input_shape=(1, 28, 28),
                num_classes=num_contents,
                nonlinear_classifier=nonlinear_classifier,
                lr=lr,
                weight_decay=weight_decay)
    else:
        if args.model == "CNN":
            model = CNN(
                data="RMNIST",
                num_domains=num_domains,
                num_contents=num_contents,
                latent_size=latent_size,
                lr=lr,
                depth=depth, 
                out_channels=out_channels,
                kernel_size=kernel_size,
                activation=activation,
                downsampling=downsampling,
                dropout=dropout,
                batch_norm=batch_norm,
                initialize=True)
        if args.model == "ERM":
            model = ERM(
                input_shape=(1, 28, 28),
                num_classes=num_contents,
                nonlinear_classifier=nonlinear_classifier,
                lr=lr,
                weight_decay=weight_decay)
    
    # Callbacks
    callbacks = [
        ClassificationLogger(args.output_dir, log_dm, train_batch, val_batch, domains, contents, images_on_val=True),
        pl.callbacks.ModelCheckpoint(monitor="val_loss", save_last=True),
    ]
    if args.model not in ["AAE", "AAE_v2", "GAN"]:
        callbacks.append(pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_epoch_start=5))

    # Trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        strategy="dp",
        precision=16,
        default_root_dir=args.output_dir,
        logger=pl.loggers.TensorBoardLogger(save_dir=os.getcwd(),
                                            name=args.output_dir),
        callbacks=callbacks,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        max_epochs=args.max_epochs,
        enable_checkpointing= not args.disable_checkpointing,
        log_every_n_steps=args.log_every_n_steps
    )

    # Main
    trainer.logger.log_hyperparams(model.hyper_param_dict)
    print(model)
    trainer.fit(model, dm)
