from argparse import ArgumentParser
import os
from tkinter import Image
import pytorch_lightning as pl
import torch

# Own Modules
from datasets.pacs import PACSDataModule
from models.cvae import CVAE
from models.ae import AE
from models.pl_ae import PL_AE
from callbacks.logger import Logger


if __name__ == "__main__":
    # Parser
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--datadir", type=str, default="data")
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
    parser.add_argument("--max_epochs", type=int, default=1000)
    
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
    contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
    batch_size = args.batch_size
    dm = PACSDataModule(root=args.datadir, domains=domains, contents=contents,
                        batch_size=batch_size, num_workers=args.num_workers)
    log_dm = PACSDataModule(root=args.datadir, domains=domains, contents=contents,
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
        if args.model == "PL_AE":
            model = PL_AE.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr)
    else:
        if args.model == "CVAE":
            model = CVAE(num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lamb=lamb, lr=lr)
        if args.model == "AE":
            model = AE(num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr)
        if args.model == "PL_AE":
            model = PL_AE(num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr)

    # Callbacks
    log_dm.setup()
    train_batch = next(iter(log_dm.train_dataloader()))
    val_batch = next(iter(log_dm.val_dataloader()))
    callbacks = [
        Logger(args.output_dir, train_batch, val_batch),
        # pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_epoch_start=5)
    ]

    # Trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        strategy="dp",
        precision=16,
        default_root_dir=args.output_dir,
        logger=pl.loggers.TensorBoardLogger(save_dir=os.getcwd(),
                                            name=args.output_dir),
        callbacks=callbacks,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
        max_epochs=args.max_epochs
    )

    # Main
    trainer.fit(model, dm)
