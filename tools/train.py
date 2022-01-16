from argparse import ArgumentParser
import os
from tkinter import Image
import pytorch_lightning as pl
import torch

# Own Modules
from datasets.pacs import PACSDataModule
from models.cvae import CVAE
from models.ae import AE
from callbacks.image_logger import ImageLogger


if __name__ == "__main__":
    # Parser
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=20)
    # Model
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--lamb", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-03)
    parser.add_argument("--ckpt_path", type=str, default=None)
    # Training
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    # Configuration
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    pl.seed_everything(17, workers=True)
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
    domains = ["art_painting", "cartoon", "photo"]
    contents = ["dog", "elephant", "giraffe",
                "guitar", "horse", "house", "person"]
    batch_size = args.batch_size
    dm = PACSDataModule(domains=domains, contents=contents,
                        batch_size=batch_size, num_workers=args.num_workers)
    log_dm = PACSDataModule(domains=domains, contents=contents,
                        batch_size=batch_size, num_workers=args.num_workers, shuffle_all=True)

    # Model
    num_domains = len(domains)
    num_contents = len(contents)
    latent_size = args.latent_size
    lamb = args.lamb
    lr = args.lr
    if args.ckpt_path not in (None, "0"):
        if args.model == "CVAE":
            model = CVAE.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lamb=lamb, lr=lr)
        if args.model == "AE":
            model = AE.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr)
    else:
        if args.model == "CVAE":
            model = CVAE(num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lamb=lamb, lr=lr)
        if args.model == "AE":
            model = AE(num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr)

    # Callbacks
    log_dm.setup()
    train_batch = next(iter(log_dm.train_dataloader()))
    val_batch = next(iter(log_dm.val_dataloader()))
    callbacks = [ImageLogger(args.output_dir, train_batch, val_batch)]

    # Trainer
    if len(args.gpus) < 3:
        auto_lr_find = True
    else:
        auto_lr_find = False

    trainer = pl.Trainer(
        gpus=args.gpus,
        strategy="dp",
        precision=16,
        default_root_dir=args.output_dir,
        logger=pl.loggers.TensorBoardLogger(save_dir=os.getcwd(),
                                            name=args.output_dir),
        callbacks=callbacks,
        auto_lr_find=auto_lr_find
    )

    # Main
    if len(args.gpus) < 3:
        trainer.tune(model)
    trainer.fit(model, dm)
