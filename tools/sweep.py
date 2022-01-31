from argparse import ArgumentParser
import os
from tkinter import Image
from xml.etree.ElementPath import prepare_child
import pytorch_lightning as pl
import torch
import copy
import random

# Own Modules
from datasets.pacs import PACSDataModule
from datasets.pacs_balanced import BalancedPACSDataModule
from models.cvae import CVAE
from models.ae import AE
from models.ae_v2 import AE_v2
from models.ae_v3 import AE_v3
from models.dccvae import DCCVAE
from models.trvae import trVAE
from models.mmd_cvae import MMD_CVAE
from models.aae import AAE
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
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--restart", action="store_true", default=False)
    parser.add_argument("--gpus", type=str, default="3,")
    parser.add_argument("--max_epochs", type=int, default=25)
    parser.add_argument("--iov", type=int, default=1)
    args = parser.parse_args()
    #################### EDIT THIS IN ORDER TO CHANGE THE SWEEP
    configs = {
        "AE_v2": {
            "latent_size": [128, 512],
        },
        "AE": {
            "latent_size": [128, 512],
        },
        "CVAE": {
            "latent_size": [128, 512],
            "lamb": [0.01, 100],
        },
        "AE_v3": {
            "kernel_size": [3, 5],
            "activation": ["selu", "elu", "relu"],
            "loss_mode": ["l1", "l2"]
        },
        "DCCVAE": {
            "feature_size": [32, 64],
            "lamb": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e+1],
        },
        "trVAE": {
            "feature_size": [32, 64],
            "mmd_size": [512, 1024],
            "dropout_rate": [0.0, 0.5],
            "lamb": [1e-5, 1e-3, 1e-1, 1.0, 1e+1],
            "beta": [1e-5, 1e-3, 1e-1, 1.0, 1e+1],
            },
        "MMD_CVAE": {
            "lamb": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e+1],
            "beta": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e+1],
            "out_channels": ["256,256,512,512,1024,1024"],
            "loss_mode": ["mmd"],
            "depth": [2],
            "activation": ["selu"],
            "latent_size":  [512],
            "lr":  [1e-4],
            "kernel_size": [3],
            "downsampling": ["stride"],
            "upsampling": ["upsample"],
            "dropout": [False],
            },
        "AAE": {
            "latent_size": [128, 256, 512],
            "depth": [1, 2],
            "batch_norm": [True, False],
            "out_channels": [
                "128,128,256,256,512,512,1024",
                "256,256,512,512,1024,1024,2048"
            ],
            "activation": ["selu", "elu"],
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
    domains = ["art_painting", "cartoon", "photo"]
    contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
    batch_size = 4
    dm = BalancedPACSDataModule(root="data/variants/PACS_small", domains=domains, contents=contents,
                        batch_size=batch_size, num_workers=20)
    log_dm = BalancedPACSDataModule(root="data/variants/PACS_small", domains=domains, contents=contents,
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
                        log_dir = f"logs/sweep/{model_name}"

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
                        if "beta" in conf:
                            beta = conf["beta"]
                            beta_string = "{:f}".format(beta)
                            beta_string = beta_string.replace(".", "-")
                            if "beta" in names:
                                log_dir += f"_{beta_string}"
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

                            

                        if args.restart or not os.path.isdir(f"{log_dir}"):
                            # Configuration
                            os.makedirs(log_dir, exist_ok=True)
                            iov = args.iov == 1
                            print(f"Images on val: {iov}")
                            callbacks = [
                                Logger(log_dir, train_batch, val_batch, domains, contents, images_on_val=iov)
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
                            if model_name == "CVAE":
                                model = CVAE(num_domains=num_domains, num_contents=num_contents,
                                            latent_size=latent_size, lamb=lamb, lr=lr)
                            if model_name == "AE":
                                model = AE(num_domains=num_domains, num_contents=num_contents,
                                            latent_size=latent_size, lr=lr)
                            if model_name == "AE_v2":
                                model = AE_v2(num_domains=num_domains, num_contents=num_contents,
                                            latent_size=latent_size, lr=lr)
                            if model_name == "AE_v3":
                                model = AE_v3(num_domains=num_domains, num_contents=num_contents, 
                                            latent_size=latent_size, lr=lr, depth=depth, out_channels=out_channels, 
                                            kernel_size=kernel_size, activation=activation, downsampling=downsampling, 
                                            upsampling=upsampling, dropout=dropout, batch_norm=batch_norm, loss_mode=loss_mode)
                            if model_name == "DCCVAE":
                                model = DCCVAE(num_domains=num_domains, num_contents=num_contents, lr=lr,
                                            latent_size=latent_size, feature_size=feature_size, loss_mode=loss_mode, lamb=lamb)
                            if model_name == "trVAE":
                                model = trVAE(num_domains=num_domains, num_contents=num_contents, latent_size=latent_size,
                                            feature_size=feature_size, mmd_size=mmd_size, dropout_rate=dropout_rate,
                                            lr=lr, lamb=lamb, beta=beta)
                            if model_name == "MMD_CVAE":
                                model = MMD_CVAE(
                                            num_domains=num_domains, num_contents=num_contents,
                                            latent_size=latent_size, lr=lr, depth=depth, 
                                            out_channels=out_channels, kernel_size=kernel_size, activation=activation,
                                            downsampling=downsampling, upsampling=upsampling, dropout=dropout,
                                            batch_norm=batch_norm, loss_mode=loss_mode, lamb=lamb, beta=beta)
                            if model_name == "AAE":
                                model = AAE(num_domains=num_domains, num_contents=num_contents,
                                            latent_size=latent_size, lr=lr, depth=depth, 
                                            out_channels=out_channels, kernel_size=kernel_size, activation=activation,
                                            downsampling=downsampling, upsampling=upsampling, dropout=dropout,
                                            batch_norm=batch_norm)


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
                                enable_checkpointing=False,
                                log_every_n_steps=5
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
