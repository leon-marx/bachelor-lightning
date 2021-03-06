from argparse import ArgumentParser
import os
import pytorch_lightning as pl
import torch

# Own Modules
from datasets.pacs import PACSDataModule
from datasets.pacs_balanced import BalancedPACSDataModule
from datasets.rotated_mnist import RMNISTDataModule
from datasets.rotated_mnist_balanced import BalancedRMNISTDataModule
from models.cvae import CVAE
from models.cvae_v2 import CVAE_v2
from models.cvae_v3 import CVAE_v3
from models.ae import AE
from models.ae_v2 import AE_v2
from models.ae_v3 import AE_v3
from models.dccvae import DCCVAE
from models.trvae import trVAE
from models.aae import AAE
from models.aae_v2 import AAE_v2
from models.gan import GAN
from models.mmd_cvae import MMD_CVAE
from callbacks.logger import Logger


if __name__ == "__main__":
    # Parser
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--domains", type=str, default=None)
    parser.add_argument("--unbalanced_data", action="store_true", default=False)
    # Model
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--feature_size", type=int, default=32)
    parser.add_argument("--mmd_size", type=int, default=512)
    parser.add_argument("--lamb", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-04)
    parser.add_argument("--ckpt_path", type=str, default="0")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--out_channels", type=str, default="128,256,512,512,1024,1024")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--downsampling", type=str, default="stride")
    parser.add_argument("--upsampling", type=str, default="stride")
    parser.add_argument("--dropout", action="store_true", default=False)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--batch_norm", action="store_true", default=False)
    parser.add_argument("--no_bn_last", action="store_true", default=False)
    parser.add_argument("--loss_mode", type=str, default="elbo")
    parser.add_argument("--no_strict", action="store_true", default=False)
    parser.add_argument("--net", type=str, default="vgg")
    parser.add_argument("--calibration", action="store_true", default=False)
    # Training
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--disable_checkpointing", action="store_true", default=False)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=17)
    parser.add_argument("--iov", type=str, default="1")
    parser.add_argument("--resume", type=str, default="0")

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
    iov = args.iov == "1"
    resume_flag = args.resume == "1"
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
        if args.unbalanced_data:
            dm = PACSDataModule(root="data", domains=domains, contents=contents,
                                batch_size=batch_size, num_workers=args.num_workers)
            log_dm = PACSDataModule(root="data", domains=domains, contents=contents,
                                batch_size=batch_size, num_workers=args.num_workers, shuffle_all=True)
        else:
            dm = BalancedPACSDataModule(root="data", domains=domains, contents=contents,
                                batch_size=batch_size, num_workers=args.num_workers)
            log_dm = BalancedPACSDataModule(root="data", domains=domains, contents=contents,
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
        if args.unbalanced_data:
            dm = PACSDataModule(root="data/variants/PACS_small", domains=domains, contents=contents,
                                batch_size=batch_size, num_workers=args.num_workers)
            log_dm = PACSDataModule(root="data/variants/PACS_small", domains=domains, contents=contents,
                                batch_size=batch_size, num_workers=args.num_workers, shuffle_all=True)
        else:
            dm = BalancedPACSDataModule(root="data/variants/PACS_small", domains=domains, contents=contents,
                                batch_size=batch_size, num_workers=args.num_workers)
            log_dm = BalancedPACSDataModule(root="data/variants/PACS_small", domains=domains, contents=contents,
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
        if args.unbalanced_data:
            dm = RMNISTDataModule(root="data", domains=domains, contents=contents,
                                batch_size=batch_size, num_workers=args.num_workers)
            log_dm = RMNISTDataModule(root="data", domains=domains, contents=contents,
                                batch_size=batch_size, num_workers=args.num_workers, shuffle_all=True)
        else:
            dm = BalancedRMNISTDataModule(root="data", domains=domains, contents=contents,
                                batch_size=batch_size, num_workers=args.num_workers)
            log_dm = BalancedRMNISTDataModule(root="data", domains=domains, contents=contents,
                                batch_size=batch_size, num_workers=args.num_workers, shuffle_all=True)
    log_dm.setup()
    train_batch = next(iter(log_dm.train_dataloader()))
    val_batch = next(iter(log_dm.val_dataloader()))

    # Model
    num_domains = len(domains)
    num_contents = len(contents)
    latent_size = args.latent_size
    feature_size = args.feature_size
    mmd_size = args.mmd_size
    lamb = args.lamb
    beta = args.beta
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
    upsampling = args.upsampling
    dropout = args.dropout
    dropout_rate = args.dropout_rate
    batch_norm = args.batch_norm
    loss_mode = args.loss_mode
    no_bn_last = args.no_bn_last
    net = args.net
    calibration = args.calibration
    data = args.data
    if args.ckpt_path != "0":
        if args.model == "CVAE":
            model = CVAE.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lamb=lamb, lr=lr, strict = not args.no_strict)
        if args.model == "CVAE_v2":
            model = CVAE_v2.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr, depth=depth, out_channels=out_channels,
                        kernel_size=kernel_size, activation=activation, downsampling=downsampling,
                        upsampling=upsampling, dropout=dropout, batch_norm=batch_norm, loss_mode=loss_mode,
                        lamb=lamb, no_bn_last=no_bn_last, strict = not args.no_strict)
        if args.model == "CVAE_v3":
            model = CVAE_v3.load_from_checkpoint(args.ckpt_path, data=data, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr, depth=depth, out_channels=out_channels,
                        kernel_size=kernel_size, activation=activation, downsampling=downsampling,
                        upsampling=upsampling, dropout=dropout, batch_norm=batch_norm, loss_mode=loss_mode,
                        lamb=lamb, no_bn_last=no_bn_last, strict = not args.no_strict)
        if args.model == "AE":
            model = AE.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr, strict = not args.no_strict)
        if args.model == "AE_v2":
            model = AE_v2.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr, strict = not args.no_strict)
        if args.model == "AE_v3":
            model = AE_v3.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr, depth=depth, out_channels=out_channels,
                        kernel_size=kernel_size, activation=activation, downsampling=downsampling,
                        upsampling=upsampling, dropout=dropout, batch_norm=batch_norm, loss_mode=loss_mode, strict = not args.no_strict)
        if args.model == "DCCVAE":
            model = DCCVAE.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents, lr=lr,
                        latent_size=latent_size, feature_size=feature_size, loss_mode=loss_mode, lamb=lamb, strict = not args.no_strict)
        if args.model == "trVAE":
            model = trVAE.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents, latent_size=latent_size,
                        feature_size=feature_size, mmd_size=mmd_size, dropout_rate=dropout_rate,
                        lr=lr, lamb=lamb, beta=beta, strict = not args.no_strict)
        if args.model == "AAE":
            if args.no_strict:
                model = AAE(data=data, num_domains=num_domains, num_contents=num_contents,
                            latent_size=latent_size, lr=lr, depth=depth,
                            out_channels=out_channels, kernel_size=kernel_size, activation=activation,
                            downsampling=downsampling, upsampling=upsampling, dropout=dropout, loss_mode=loss_mode,
                            batch_norm=batch_norm, initialize=True)
                current_model_dict = model.state_dict()
                loaded_state_dict = torch.load(args.ckpt_path, map_location=f"cuda:{args.gpus[0]}")["state_dict"]
                new_state_dict={k:v.cpu() if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
                model.load_state_dict(new_state_dict, strict=False)
                loaded_state_dict = None
                new_state_dict = None
            else:
                model = AAE.load_from_checkpoint(args.ckpt_path, data=data, num_domains=num_domains, num_contents=num_contents,
                            latent_size=latent_size, lr=lr, depth=depth,
                            out_channels=out_channels, kernel_size=kernel_size, activation=activation,
                            downsampling=downsampling, upsampling=upsampling, dropout=dropout, loss_mode=loss_mode,
                            batch_norm=batch_norm, strict = not args.no_strict)
        if args.model == "AAE_v2":
            if args.no_strict:
                model = AAE_v2(data=data, num_domains=num_domains, num_contents=num_contents,
                            latent_size=latent_size, lr=lr, depth=depth,
                            out_channels=out_channels, kernel_size=kernel_size, activation=activation,
                            downsampling=downsampling, upsampling=upsampling, dropout=dropout, loss_mode=loss_mode,
                            lamb=lamb, net=net, calibration=calibration, batch_norm=batch_norm, initialize=True)
                current_model_dict = model.state_dict()
                loaded_state_dict = torch.load(args.ckpt_path, map_location=f"cuda:{args.gpus[0]}")["state_dict"]
                new_state_dict={k:v.cpu() if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
                model.load_state_dict(new_state_dict, strict=False)
                loaded_state_dict = None
                new_state_dict = None
            else:
                model = AAE_v2.load_from_checkpoint(args.ckpt_path, data=data, num_domains=num_domains, num_contents=num_contents,
                            latent_size=latent_size, lr=lr, depth=depth,
                            out_channels=out_channels, kernel_size=kernel_size, activation=activation,
                            downsampling=downsampling, upsampling=upsampling, dropout=dropout, loss_mode=loss_mode,
                            lamb=lamb, net=net, calibration=calibration, batch_norm=batch_norm, strict = not args.no_strict)
        if args.model == "GAN":
            model = GAN.load_from_checkpoint(args.ckpt_path, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr, depth=depth,
                        out_channels=out_channels, kernel_size=kernel_size, activation=activation,
                        downsampling=downsampling, upsampling=upsampling, dropout=dropout,
                        batch_norm=batch_norm, strict = not args.no_strict)
        if args.model == "MMD_CVAE":
            model = MMD_CVAE.load_from_checkpoint(args.ckpt_path, data=data, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr, depth=depth,
                        out_channels=out_channels, kernel_size=kernel_size, activation=activation,
                        downsampling=downsampling, upsampling=upsampling, dropout=dropout,
                        batch_norm=batch_norm, loss_mode=loss_mode, lamb=lamb, beta=beta, strict = not args.no_strict)
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
            model = CVAE_v3(data=data, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr, depth=depth, out_channels=out_channels,
                        kernel_size=kernel_size, activation=activation, downsampling=downsampling,
                        upsampling=upsampling, dropout=dropout, batch_norm=batch_norm, loss_mode=loss_mode,
                        lamb=lamb, no_bn_last=no_bn_last, initialize=True)
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
        if args.model == "DCCVAE":
            model = DCCVAE(num_domains=num_domains, num_contents=num_contents, lr=lr,
                        latent_size=latent_size, feature_size=feature_size, loss_mode=loss_mode, lamb=lamb)
        if args.model == "trVAE":
            model = trVAE(num_domains=num_domains, num_contents=num_contents, latent_size=latent_size,
                        feature_size=feature_size, mmd_size=mmd_size, dropout_rate=dropout_rate,
                        lr=lr, lamb=lamb, beta=beta)
        if args.model == "AAE":
            model = AAE(data=data, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr, depth=depth,
                        out_channels=out_channels, kernel_size=kernel_size, activation=activation,
                        downsampling=downsampling, upsampling=upsampling, dropout=dropout, loss_mode=loss_mode,
                        batch_norm=batch_norm, initialize=True)
        if args.model == "AAE_v2":
            model = AAE_v2(data=data, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr, depth=depth,
                        out_channels=out_channels, kernel_size=kernel_size, activation=activation,
                        downsampling=downsampling, upsampling=upsampling, dropout=dropout, loss_mode=loss_mode,
                        lamb=lamb, net=net, calibration=calibration, batch_norm=batch_norm, initialize=True)
        if args.model == "GAN":
            model = GAN(num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr, depth=depth,
                        out_channels=out_channels, kernel_size=kernel_size, activation=activation,
                        downsampling=downsampling, upsampling=upsampling, dropout=dropout,
                        batch_norm=batch_norm)
        if args.model == "MMD_CVAE":
            model = MMD_CVAE(data=data, num_domains=num_domains, num_contents=num_contents,
                        latent_size=latent_size, lr=lr, depth=depth,
                        out_channels=out_channels, kernel_size=kernel_size, activation=activation,
                        downsampling=downsampling, upsampling=upsampling, dropout=dropout,
                        batch_norm=batch_norm, loss_mode=loss_mode, lamb=lamb, beta=beta, initialize=True)

    # Callbacks
    callbacks = [
        Logger(args.output_dir, log_dm, train_batch, val_batch, domains, contents, images_on_val=iov),
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
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
        max_epochs=args.max_epochs,
        enable_checkpointing= not args.disable_checkpointing,
        log_every_n_steps=args.log_every_n_steps
    )

    # Main
    trainer.logger.log_hyperparams(model.hyper_param_dict)
    print(model)
    if resume_flag:
        trainer.fit(model, dm, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, dm)
