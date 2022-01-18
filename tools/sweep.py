from argparse import ArgumentParser
import os
from tkinter import Image
import pytorch_lightning as pl
import torch

# Own Modules
from datasets.pacs import PACSDataModule
from models.cvae import CVAE
from models.ae import AE
from models.ae_v2 import AE_v2
from models.ae_v3 import AE_v3
from callbacks.logger import Logger


def get_combinations(arg_dict):
    combinations = [arg_dict]
    for key, value in arg_dict.items():
        new_combs = []
        for dd in combinations:
            if isinstance(value, list):
                for val in value:
                    d = copy.deepcopy(dd)
                    d[key] = val
                    new_combs.append(d)
        combinations = new_combs
    return combinations


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--start_step", type=int, default=0)
    args = parser.parse_args()
    #################### EDIT THIS IN ORDER TO CHANGE THE SWEEP
    configs = {
        "AE_v3": {
            "out_channels": [
                "128,256,512,512,1024,1024",
                "512,512,512,512,512,512",
                "128,128,256,256,512,512",
                "128,256,512,512,1024,2048"
            ],
            "latent_size": [128, 512],
            "depth": [1, 2],
            "kernel_size": [3, 5],
            "activation": ["gelu", "lrelu"],
            "downsampling": ["stride", "maxpool"],
            "upsampling": ["stride", "upsample"],
            "dropout": [True, False],
            "batch_norm": [True, False],
        },
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
    batch_size = 8
    dm = PACSDataModule(root="data/variants/PACS_small", domains=domains, contents=contents,
                        batch_size=batch_size, num_workers=20)
    log_dm = PACSDataModule(root="data/variants/PACS_small", domains=domains, contents=contents,
                        batch_size=batch_size, num_workers=20, shuffle_all=True)
    num_domains = len(domains)
    num_contents = len(contents)

    # Callbacks
    log_dm.setup()
    train_batch = next(iter(log_dm.train_dataloader()))
    val_batch = next(iter(log_dm.val_dataloader()))

    step = 0
    for model in configs:
        print(f"Starting loop over {model} configurations.")
        combinations = get_combinations(configs[model])
        for conf in combinations:
            if step >= args.start_step:
                print(f"Configuration: {conf}")
                # Default values
                log_dir = f"logs/sweep/{model}"
                latent_size =  512
                lamb =  10
                lr =  1e-4
                depth = 2
                out_channels = "128,256,512,512,1024,1024"
                kernel_size = 3
                activation = "relu"
                downsampling = "stride"
                upsampling = "stride"
                dropout = False
                batch_norm = False


                if "latent_size" in conf:
                    latent_size = conf["latent_size"]
                    log_dir += f"_{latent_size}"
                if "lamb" in conf:
                    lamb = conf["lamb"]
                    log_dir += f"_{lamb}"
                if "lr" in conf:
                    lr = conf["lr"]
                    lr_string = "{:e}".format(lr)
                    log_dir += f"_{lr_string}"
                if "depth" in conf:
                    depth = conf["depth"]
                    log_dir += f"_{depth}"
                if "kernel_size" in conf:
                    kernel_size = conf["kernel_size"]
                    log_dir += f"_{kernel_size}"
                if "activation" in conf:
                    activation = conf["activation"]
                    log_dir += f"_{activation}"
                if "downsampling" in conf:
                    downsampling = conf["downsampling"]
                    log_dir += f"_{downsampling}"
                if "upsampling" in conf:
                    upsampling = conf["upsampling"]
                    log_dir += f"_{upsampling}"
                if "dropout" in conf:
                    dropout = conf["dropout"]
                    log_dir += f"_{dropout}"
                if "batch_norm" in conf:
                    batch_norm = conf["batch_norm"]
                    log_dir += f"_{batch_norm}"
                if "out_channels" in conf:
                    out_channels = conf["out_channels"]
                    log_dir += f"_{out_channels}"
                    

                # Configuration
                if log_dir is not None:
                    os.makedirs(log_dir, exist_ok=True)
                callbacks = [
                    Logger(log_dir, train_batch, val_batch)
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
                if model == "CVAE":
                    model = CVAE(num_domains=num_domains, num_contents=num_contents,
                                latent_size=latent_size, lamb=lamb, lr=lr)
                if model == "AE":
                    model = AE(num_domains=num_domains, num_contents=num_contents,
                                latent_size=latent_size, lr=lr)
                if model == "AE_v2":
                    model = AE_v2(num_domains=num_domains, num_contents=num_contents,
                                latent_size=latent_size, lr=lr)
                if model == "AE_v3":
                    model = AE_v3(num_domains=num_domains, num_contents=num_contents, 
                                latent_size=latent_size, lr=lr, depth=depth, out_channels=out_channels, 
                                kernel_size=kernel_size, activation=activation, downsampling=downsampling, 
                                upsampling=upsampling, dropout=dropout, batch_norm=batch_norm)


                # Trainer
                trainer = pl.Trainer(
                    gpus="3,2",
                    strategy="dp",
                    precision=16,
                    default_root_dir=log_dir,
                    logger=pl.loggers.TensorBoardLogger(save_dir=os.getcwd(),
                                                        name=log_dir),
                    callbacks=callbacks,
                    gradient_clip_val=0.5,
                    gradient_clip_algorithm="value",
                    max_epochs=25,
                    enable_checkpointing=False
                )

                # Main
                trainer.fit(model, dm)

                print(f"Completed step {step}!")
            step += 1
