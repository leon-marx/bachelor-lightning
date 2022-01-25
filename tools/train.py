from argparse import ArgumentParser
import os
import pytorch_lightning as pl
import torch

# Own Modules
from datasets.pacs import PACSDataModule
from models.cvae import CVAE
from models.cvae_v2 import CVAE_v2
from models.cvae_v3 import CVAE_v3
from models.cvae_v4 import CVAE_v4
from models.ae import AE
from models.ae_v2 import AE_v2
from models.ae_v3 import AE_v3
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
    parser.add_argument("--lr", type=float, default=1e-04)
    parser.add_argument("--ckpt_path", type=str, default="0")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--out_channels", type=str, default="128,256,512,512,1024,1024")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--downsampling", type=str, default="stride")
    parser.add_argument("--upsampling", type=str, default="stride")
    parser.add_argument("--dropout", action="store_true", default=False)
    parser.add_argument("--batch_norm", action="store_true", default=False)
    parser.add_argument("--no_bn_last", action="store_true", default=False)
    parser.add_argument("--loss_mode", type=str, default="l2")
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--end_level", type=int, default=8)
    # Training
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--enable_checkpointing", action="store_true", default=True)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    
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
    depth = args.depth
    out_channels = list(map(int, args.out_channels.split(",")))
    kernel_size = args.kernel_size
    activation = args.activation
    if activation == "relu":
        activation = torch.nn.ReLU()
    if activation == "gelu":
        activation = torch.nn.GELU()
    if activation == "lrelu":
        activation = torch.nn.LeakyReLU()
    if activation == "elu":
        activation = torch.nn.ELU()
    if activation == "selu":
        activation = torch.nn.SELU()
    downsampling = args.downsampling
    upsampling = args.upsampling
    dropout = args.dropout
    batch_norm = args.batch_norm
    loss_mode = args.loss_mode
    no_bn_last = args.no_bn_last
    level = args.level
    end_level = args.end_level
    if args.ckpt_path != "0":
        if args.model == "CVAE":
            model = CVAE.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lamb=lamb, lr=lr)
        if args.model == "CVAE_v2":
            model = CVAE_v2.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents, 
                        latent_size=latent_size, lr=lr, depth=depth, out_channels=out_channels, 
                        kernel_size=kernel_size, activation=activation, downsampling=downsampling, 
                        upsampling=upsampling, dropout=dropout, batch_norm=batch_norm, loss_mode=loss_mode,
                        lamb=lamb, no_bn_last=no_bn_last, strict=not no_bn_last)
        if args.model == "CVAE_v3":
            model = CVAE_v3.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents, 
                        latent_size=latent_size, lr=lr, depth=depth, out_channels=out_channels, 
                        kernel_size=kernel_size, activation=activation, downsampling=downsampling, 
                        upsampling=upsampling, dropout=dropout, batch_norm=batch_norm, loss_mode=loss_mode,
                        lamb=lamb, no_bn_last=no_bn_last, strict=not no_bn_last)
        if args.model == "CVAE_v4":
            model = CVAE_v4.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents, 
                        lr=lr, depth=depth, out_channels=out_channels, 
                        kernel_size=kernel_size, activation=activation, downsampling=downsampling, 
                        upsampling=upsampling, dropout=dropout, batch_norm=batch_norm, loss_mode=loss_mode,
                        lamb=lamb, level=level, no_bn_last=no_bn_last, strict=not no_bn_last)
        if args.model == "AE":
            model = AE.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr)
        if args.model == "AE_v2":
            model = AE_v2.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr)
        if args.model == "AE_v3":
            model = AE_v3.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents, 
                        latent_size=latent_size, lr=lr, depth=depth, out_channels=out_channels, 
                        kernel_size=kernel_size, activation=activation, downsampling=downsampling, 
                        upsampling=upsampling, dropout=dropout, batch_norm=batch_norm, loss_mode=loss_mode, strict=not no_bn_last)
    else:
        if args.model == "CVAE":
            model = CVAE(num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lamb=lamb, lr=lr)
        if args.model == "CVAE_v2":
            model = CVAE_v2(num_domains=num_domains, num_contents=num_contents, 
                        latent_size=latent_size, lr=lr, depth=depth, out_channels=out_channels, 
                        kernel_size=kernel_size, activation=activation, downsampling=downsampling, 
                        upsampling=upsampling, dropout=dropout, batch_norm=batch_norm, loss_mode=loss_mode,
                        lamb=lamb, no_bn_last=no_bn_last)
        if args.model == "CVAE_v3":
            model = CVAE_v3(num_domains=num_domains, num_contents=num_contents, 
                        latent_size=latent_size, lr=lr, depth=depth, out_channels=out_channels, 
                        kernel_size=kernel_size, activation=activation, downsampling=downsampling, 
                        upsampling=upsampling, dropout=dropout, batch_norm=batch_norm, loss_mode=loss_mode,
                        lamb=lamb, no_bn_last=no_bn_last)
        if args.model == "CVAE_v4":
            model = CVAE_v4(num_domains=num_domains, num_contents=num_contents, 
                        lr=lr, depth=depth, out_channels=out_channels, 
                        kernel_size=kernel_size, activation=activation, downsampling=downsampling, 
                        upsampling=upsampling, dropout=dropout, batch_norm=batch_norm, loss_mode=loss_mode,
                        lamb=lamb, level=level, no_bn_last=no_bn_last)
        if args.model == "AE":
            model = AE(num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr)
        if args.model == "AE_v2":
            model = AE_v2(num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr)
        if args.model == "AE_v3":
            model = AE_v3(num_domains=num_domains, num_contents=num_contents, 
                        latent_size=latent_size, lr=lr, depth=depth, out_channels=out_channels, 
                        kernel_size=kernel_size, activation=activation, downsampling=downsampling, 
                        upsampling=upsampling, dropout=dropout, batch_norm=batch_norm, loss_mode=loss_mode)

    # Main
    log_dm.setup()
    train_batch = next(iter(log_dm.train_dataloader()))
    val_batch = next(iter(log_dm.val_dataloader()))

    if level is None:
        # Callbacks
        callbacks = [
            Logger(args.output_dir, train_batch, val_batch, images_on_val=True),
            pl.callbacks.ModelCheckpoint(monitor="val_loss"),
            pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_epoch_start=5)
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
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            max_epochs=args.max_epochs,
            enable_checkpointing=args.enable_checkpointing,
            log_every_n_steps=args.log_every_n_steps
        )
        if args.model in ["AE_v3", "CVAE_v2", "CVAE_v3", "CVAE_v4"]:
            trainer.logger.log_hyperparams(model.hyper_param_dict)
            print(model)
        trainer.fit(model, dm)

    else:
        if args.model in ["AE_v3", "CVAE_v2", "CVAE_v3", "CVAE_v4"]:
            print(model)
    try:
        for lvl in range(level, end_level+1, 1):
            print("")
            print(f"Starting training on level {lvl}:")
            # Callbacks
            callbacks = [
                Logger(args.output_dir, train_batch, val_batch, images_on_val=True),
                pl.callbacks.ModelCheckpoint(monitor="val_loss"),
                pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_epoch_start=5)
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
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
                max_epochs=10, # 50
                enable_checkpointing=args.enable_checkpointing,
                log_every_n_steps=10 # 10
            )
            model.set_level(lvl)
            trainer.logger.log_hyperparams(model.hyper_param_dict)
            trainer.fit(model, dm)
    except KeyboardInterrupt:
        print("Interrupting training!")
        