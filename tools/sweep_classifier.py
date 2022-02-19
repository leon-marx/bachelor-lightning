from argparse import ArgumentParser
import os
from tkinter import Image
import pytorch_lightning as pl
import torch
import copy
import random

# Own Modules
from datasets.pacs import PACSDataModule
from datasets.rotated_mnist import RMNISTDataModule
from models.erm import ERM
from models.cnn import CNN
from callbacks.classification_logger import ClassificationLogger


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
        "CNN": {
            "data": ["RMNIST"],
            "num_domains": [5],
            "num_contents": [10],
            "latent_size": [128],
            "lr": [1e-4],
            "depth": [2],
            "out_channels": ["128,128,256,256,512,512"],
            "kernel_size": [3],
            "activation": ["relu"],
            "downsampling": ["stride"],
            "dropout": [False],
            "batch_norm": [True],
            "initialize": [True],
            "domains": ["01234", "01235", "01245", "01345", "02345", "12345"],
            "root": ["data", "data/variants/RMNIST_augmented"]
            },
        "ERM": {
            "input_shape": [(1, 28, 28)],
            "nonlinear_classifier": [False],
            "lr": [1e-4],
            "weight_decay": [0.0],
            "domains": ["01234", "01235", "01245", "01345", "02345", "12345"],
            "root": ["data", "data/variants/RMNIST_augmented"]
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

    if args.data == "PACS":
        domains = ["art_painting", "cartoon", "photo"]
    domain_dict = {
            "0": 0,
            "1": 15,
            "2": 30,
            "3": 45,
            "4": 60,
            "5": 75,
    }

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
                        if "downsampling" in conf:
                            downsampling = conf["downsampling"]
                            if "downsampling" in names:
                                log_dir += f"_{downsampling}"
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
                            out_channels = list(map(int, out_channels.split(",")))
                        if "initialize" in conf:
                            initialize = conf["initialize"]
                            if "initialize" in names:
                                log_dir += f"_{initialize}"
                        if "nonlinear_classifier" in conf:
                            nonlinear_classifier = conf["nonlinear_classifier"]
                            if "nonlinear_classifier" in names:
                                log_dir += f"_{nonlinear_classifier}"
                        if "weight_decay" in conf:
                            weight_decay = conf["weight_decay"]
                            if "weight_decay" in names:
                                log_dir += f"_{weight_decay}"
                        if "input_shape" in conf:
                            input_shape = conf["input_shape"]
                        if "data" in conf:
                            data = conf["data"]
                            if "data" in names:
                                log_dir += f"_{data}"
                        if "domains" in conf:
                            domains = [domain_dict[d] for d in conf["domains"]]
                            if "domains" in names:
                                log_dir += f"_{conf['domains']}"
                        if "root" in conf:
                            root = conf["root"]
                            if "root" in names:
                                if "augmented" in conf["root"]:
                                    rootlog = "augmented"
                                else:
                                    rootlog = "normal"
                                log_dir += f"_{rootlog}"

                        # Dataset
                        batch_size = args.batch_size
                        if args.data == "PACS":
                            domains = ["art_painting", "cartoon", "photo"]
                            contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
                            dm = PACSDataModule(root="data/variants/PACS_small", domains=domains, contents=contents,
                                                batch_size=batch_size, num_workers=20)
                            log_dm = PACSDataModule(root="data/variants/PACS_small", domains=domains, contents=contents,
                                                batch_size=batch_size, num_workers=20, shuffle_all=True)
                        elif args.data == "RMNIST":
                            contents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                            dm = RMNISTDataModule(root=root, domains=domains, contents=contents,
                                                batch_size=batch_size, num_workers=20)
                            log_dm = RMNISTDataModule(root=root, domains=domains, contents=contents,
                                                batch_size=batch_size, num_workers=20, shuffle_all=True)
                        num_domains = len(domains)
                        num_contents = len(contents)

                        # Callbacks
                        log_dm.setup()
                        train_batch = next(iter(log_dm.train_dataloader()))
                        val_batch = next(iter(log_dm.val_dataloader()))



                        if args.restart or not os.path.isdir(f"{log_dir}"):
                            # Configuration
                            os.makedirs(log_dir, exist_ok=True)
                            iov = args.iov == 1
                            print(f"Images on val: {iov}")
                            callbacks = [
                                ClassificationLogger(log_dir, log_dm, train_batch, val_batch, domains, contents, images_on_val=iov)
                            ]

                            print("Args:")
                            for k, v in sorted(conf.items()):
                                print(f"    {k}: {v}")

                            # Model
                            if model_name == "CNN":
                                model = CNN(
                                    data=data,
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
                                    initialize=initialize)
                            if model_name == "ERM":
                                model = ERM(
                                    input_shape=input_shape,
                                    num_classes=num_contents,
                                    nonlinear_classifier=nonlinear_classifier,
                                    lr=lr,
                                    weight_decay=weight_decay)

                            # Trainer
                            trainer = pl.Trainer(
                                gpus=args.gpus,
                                strategy="dp",
                                precision=16,
                                default_root_dir=log_dir,
                                logger=pl.loggers.TensorBoardLogger(save_dir=os.getcwd(),
                                                                    name=log_dir),
                                callbacks=callbacks,
                                gradient_clip_val=0.5,
                                gradient_clip_algorithm="value",
                                max_epochs=args.max_epochs,
                                enable_checkpointing=False,
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
