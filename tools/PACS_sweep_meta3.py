from argparse import ArgumentParser
import os
from tkinter import Image
import pytorch_lightning as pl
import torch
import copy
import random

# Own Modules
from datasets.pacs import PACSDataModule
from datasets.pacs_balanced import BalancedPACSDataModule
from datasets.rotated_mnist import RMNISTDataModule
from datasets.rotated_mnist_balanced import BalancedRMNISTDataModule
from models.cvae import CVAE
from models.cvae_v3 import CVAE_v3
from models.ae import AE
from models.ae_v2 import AE_v2
from models.ae_v3 import AE_v3
from models.dccvae import DCCVAE
from models.trvae import trVAE
from models.mmd_cvae import MMD_CVAE
from models.aae import AAE
from models.aae_v2 import AAE_v2
from callbacks.logger import Logger


def get_combinations(arg_dict):
    combinations = [arg_dict]
    names = []
    for key, value in arg_dict.items():
        new_combs = []
        for dd in combinations:
            if isinstance(value, list):
                if len(value) > 1 and key not in names:
                    names.append(key)
                for val in value:
                    d = copy.deepcopy(dd)
                    d[key] = val
                    new_combs.append(d)
        if new_combs != []:
            combinations = new_combs
    return combinations, names


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--log_name", type=str, default=None)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--restart", action="store_true", default=False)
    parser.add_argument("--test_mode", action="store_true", default=False)
    parser.add_argument("--gpus", type=str, default="3,")
    parser.add_argument("--max_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--iov", type=int, default=1)
    args = parser.parse_args()
    #################### EDIT THIS IN ORDER TO CHANGE THE SWEEP
    configs = {
        "CVAE_v3": {
            "data": ["PACS"],
            "num_domains": [3],
            "num_contents": [7],
            "latent_size": [1024],
            "lr": [1e-5],
            "depth": [1],
            "out_channels": ["256,256,512,512,1024,1024"],
            "kernel_size": [3],
            "activation": ["elu"],
            "downsampling": ["maxpool"],
            "upsampling": ["upsample"],
            "dropout": [False],
            "batch_norm": [True],
            "loss_mode": ["elbo"],
            "lamb": [1e-4],
            "max_lamb": [0.01, 1.0],
            "no_bn_last": [True],
            "initialize": [True]
            },
        "MMD_CVAE": {
            "data": ["PACS"],
            "num_domains": [3],
            "num_contents": [7],
            "latent_size": [1024],
            "lr": [1e-5],
            "depth": [1],
            "out_channels": ["256,256,512,512,1024,1024"],
            "kernel_size": [3],
            "activation": ["elu"],
            "downsampling": ["maxpool"],
            "upsampling": ["upsample"],
            "dropout": [False],
            "batch_norm": [True],
            "loss_mode": ["mmd"],
            "lamb": [1e-4],
            "beta": [1e-4],
            "max_lamb": [0.01, 0.1, 1.0],
            "max_beta": [0.01, 0.1, 1.0],
            "no_bn_last": [True],
            "initialize": [True]
            },
        "AAE_v2": {
            "data": ["PACS"],
            "num_domains": [3],
            "num_contents": [7],
            "latent_size": [1024],
            "lr": [1e-5],
            "depth": [1],
            "out_channels": ["256,256,512,512,1024,1024,1024"],
            "kernel_size": [3],
            "activation": ["elu"],
            "downsampling": ["maxpool"],
            "upsampling": ["upsample"],
            "dropout": [False],
            "batch_norm": [True],
            "loss_mode": ["l2"],
            "lamb": [0.0],
            "no_bn_last": [True],
            "initialize": [True],
            "net": ["vgg"],
            "calibration": [True]
            },
    }
    ####################
    # Configuration
    pl.seed_everything(17, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Printing Configuration
    print("Environment:")
    print(f"    PyTorch: {torch.__version__}")
    print(f"    CUDA: {torch.version.cuda}")
    print(f"    CUDNN: {torch.backends.cudnn.version()}")

    # Dataset
    batch_size = args.batch_size
    if args.data == "PACS":
        domains = ["art_painting", "cartoon", "photo"]
        contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        dm = BalancedPACSDataModule(root="data", domains=domains, contents=contents,
                            batch_size=batch_size, num_workers=20)
        log_dm = BalancedPACSDataModule(root="data", domains=domains, contents=contents,
                            batch_size=batch_size, num_workers=20, shuffle_all=True)
    elif args.data == "RMNIST":
        domains = [0, 15, 30, 45, 60, 75]
        contents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        dm = BalancedRMNISTDataModule(root="data", domains=domains, contents=contents,
                            batch_size=batch_size, num_workers=20)
        log_dm = BalancedRMNISTDataModule(root="data", domains=domains, contents=contents,
                            batch_size=batch_size, num_workers=20, shuffle_all=True)
    num_domains = len(domains)
    num_contents = len(contents)

    # Callbacks
    log_dm.setup()
    train_batch = next(iter(log_dm.train_dataloader()))
    val_batch = next(iter(log_dm.val_dataloader()))

    def get_precision(model_name):
        if model_name == "AEE":
            return 32
        else:
            return 16

    step = 0
    try:
        for model_name in args.models.split(","):
            try:
                print(f"Starting loop over {model_name} configurations.")
                combinations, names = get_combinations(configs[model_name])
                # random.shuffle(combinations)
                for conf in combinations:
                    try:
                        print(f"Configuration: {conf}")
                        # Default values
                        log_dir = f"logs/sweep_{args.log_name}/{model_name}"

                        if "latent_size" in conf:
                            latent_size = conf["latent_size"]
                            if "latent_size" in names:
                                log_dir += f"_{latent_size}"
                        if "feature_size" in conf:
                            feature_size = conf["feature_size"]
                            if "feature_size" in names:
                                log_dir += f"_{feature_size}"
                        if "mmd_size" in conf:
                            mmd_size = conf["mmd_size"]
                            if "mmd_size" in names:
                                log_dir += f"_{mmd_size}"
                        if "lamb" in conf:
                            lamb = conf["lamb"]
                            lamb_string = "{:f}".format(lamb)
                            lamb_string = lamb_string.replace(".", "-")
                            if "lamb" in names:
                                log_dir += f"_{lamb_string}"
                        if "max_lamb" in conf:
                            max_lamb = conf["max_lamb"]
                            max_lamb_string = "{:f}".format(max_lamb)
                            max_lamb_string = max_lamb_string.replace(".", "-")
                            if "max_lamb" in names:
                                log_dir += f"_{max_lamb_string}"
                        else:
                            max_lamb = 1.0
                        if "beta" in conf:
                            beta = conf["beta"]
                            beta_string = "{:f}".format(beta)
                            beta_string = beta_string.replace(".", "-")
                            if "beta" in names:
                                log_dir += f"_{beta_string}"
                        if "max_beta" in conf:
                            max_beta = conf["max_beta"]
                            max_beta_string = "{:f}".format(max_beta)
                            max_beta_string = max_beta_string.replace(".", "-")
                            if "max_beta" in names:
                                log_dir += f"_{max_beta_string}"
                        else:
                            max_beta = 1.0
                        if "dropout_rate" in conf:
                            dropout_rate = conf["dropout_rate"]
                            dropout_rate_string = "{:f}".format(dropout_rate)
                            dropout_rate_string = dropout_rate_string.replace(".", "-")
                            if "dropout_rate" in names:
                                log_dir += f"_{dropout_rate_string}"
                        if "lr" in conf:
                            lr = conf["lr"]
                            lr_string = "{:f}".format(lr)
                            lr_string = lr_string.replace(".", "-")
                            if "lr" in names:
                                log_dir += f"_{lr_string}"
                        if "depth" in conf:
                            depth = conf["depth"]
                            if "depth" in names:
                                log_dir += f"_{depth}"
                        if "kernel_size" in conf:
                            kernel_size = conf["kernel_size"]
                            if "kernel_size" in names:
                                log_dir += f"_{kernel_size}"
                        if "activation" in conf:
                            activation = conf["activation"]
                            if "activation" in names:
                                log_dir += f"_{activation}"
                        if "downsampling" in conf:
                            downsampling = conf["downsampling"]
                            if "downsampling" in names:
                                log_dir += f"_{downsampling}"
                        if "upsampling" in conf:
                            upsampling = conf["upsampling"]
                            if "upsampling" in names:
                                log_dir += f"_{upsampling}"
                        if "dropout" in conf:
                            dropout = conf["dropout"]
                            if "dropout" in names:
                                log_dir += f"_{dropout}"
                        if "batch_norm" in conf:
                            batch_norm = conf["batch_norm"]
                            if "batch_norm" in names:
                                log_dir += f"_{batch_norm}"
                        if "out_channels" in conf:
                            out_channels = conf["out_channels"]
                            if "out_channels" in names:
                                log_dir += f"_{out_channels.replace(',', '-')}"
                        if "loss_mode" in conf:
                            loss_mode = conf["loss_mode"]
                            if "loss_mode" in names:
                                log_dir += f"_{loss_mode}"
                        if "net" in conf:
                            net = conf["net"]
                            if "net" in names:
                                log_dir += f"_{net}"
                        if "calibration" in conf:
                            calibration = conf["calibration"]
                            if "calibration" in names:
                                log_dir += f"_{calibration}"



                        if args.restart or not os.path.isdir(f"{log_dir}"):
                            # Configuration
                            os.makedirs(log_dir, exist_ok=True)
                            iov = args.iov == 1
                            print(f"Images on val: {iov}")
                            callbacks = [
                                Logger(log_dir, log_dm, train_batch, val_batch, domains, contents, images_on_val=iov),
                                pl.callbacks.ModelCheckpoint(monitor="val_loss", save_last=True),
                            ]

                            print("Args:")
                            for k, v in sorted(conf.items()):
                                print(f"    {k}: {v}")

                            # Model
                            out_channels = list(map(int, out_channels.split(",")))
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
                            if "CVAE_v3" in model_name:
                                model = CVAE_v3(data=args.data, num_domains=num_domains, num_contents=num_contents,
                                            latent_size=latent_size, lr=lr, depth=depth,
                                            out_channels=out_channels, kernel_size=kernel_size, activation=activation,
                                            downsampling=downsampling, upsampling=upsampling, dropout=dropout,
                                            batch_norm=batch_norm, loss_mode=loss_mode, lamb=lamb, initialize=True, max_lamb=max_lamb)
                            if "MMD_CVAE" in model_name:
                                model = MMD_CVAE(data=args.data, num_domains=num_domains, num_contents=num_contents,
                                            latent_size=latent_size, lr=lr, depth=depth,
                                            out_channels=out_channels, kernel_size=kernel_size, activation=activation,
                                            downsampling=downsampling, upsampling=upsampling, dropout=dropout,
                                            batch_norm=batch_norm, loss_mode=loss_mode, lamb=lamb, beta=beta, initialize=True, max_lamb=max_lamb, max_beta=max_beta)
                            if "AAE_v2" in model_name:
                                model = AAE_v2(data=args.data, num_domains=num_domains, num_contents=num_contents,
                                            latent_size=latent_size, lr=lr, depth=depth,
                                            out_channels=out_channels, kernel_size=kernel_size, activation=activation,
                                            downsampling=downsampling, upsampling=upsampling, dropout=dropout,
                                            batch_norm=batch_norm, loss_mode=loss_mode, lamb=lamb, net=net, calibration=calibration, initialize=True)


                            # Trainer
                            trainer = pl.Trainer(
                                gpus=args.gpus,
                                strategy="dp",
                                precision=get_precision(model_name),
                                default_root_dir=log_dir,
                                logger=pl.loggers.TensorBoardLogger(save_dir=os.getcwd(),
                                                                    name=log_dir),
                                callbacks=callbacks,
                                gradient_clip_val=0.5,
                                gradient_clip_algorithm="value",
                                max_epochs=args.max_epochs,
                                enable_checkpointing=True,
                                log_every_n_steps=5,
                                fast_dev_run=args.test_mode
                            )

                            # Main
                            trainer.logger.log_hyperparams(model.hyper_param_dict)
                            trainer.fit(model, dm)
                            print(f"Completed step {step}!")
                        step += 1
                    except KeyboardInterrupt:
                        print("Interrupted the sweep!")
                        break
            except KeyboardInterrupt:
                print("Interrupted the sweep!")
                break
    except KeyboardInterrupt:
        print("Interrupted the sweep!")
