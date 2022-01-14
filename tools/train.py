from argparse import ArgumentParser
import os
import pytorch_lightning as pl
import torch

# Own Modules
from datasets.pacs import PACSDataModule
from models.cvae import CVAE


if __name__ == "__main__":
    # Parser
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=20)
    # Model
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

    # Model
    num_domains = len(domains)
    num_contents = len(contents)
    latent_size = args.latent_size
    lamb = args.lamb
    lr = args.lr
    if args.ckpt_path not in (None, "0"):
        model = CVAE.load_from_checkpoint(args.ckpt_path)
    else:
        model = CVAE(num_domains=num_domains, num_contents=num_contents,
                     latent_size=latent_size, lamb=lamb, lr=lr)

    # Trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        strategy="ddp",
        precision=16,
        default_root_dir=args.output_dir,
        logger=pl.loggers.TensorBoardLogger(save_dir=os.getcwd(),
                                            name=args.output_dir),
    )

    # Main
    trainer.fit(model, dm)


"""
from argparse import ArgumentParser
import os
import torch
import pytorch_lightning as pl

from lm.conv_cvae_lightning import LM_CCVAE_LIGHTNING
from lm.pacs_lightning import LM_PACS_LIGHTNING


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="LM_PACS_LIGHTNING")
    parser.add_argument("--data_dir", type=str, default="./../data/")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--test_env", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--checkpoint_freq", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default="LM_CCVAE_LIGHTNING")
    parser.add_argument("--latent_size", type=str, default=None)
    parser.add_argument("--lamb", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--save_best_every_checkpoint", action="store_true")

    args = parser.parse_args()

    # os.makedirs(args.output_dir, exist_ok=True)

    print("Environment:")
    # print("\tPython: {}".format(sys.version.split(" ")[0]))
    # print("\tPyTorch: {}".format(torch.__version__))
    # print("\tTorchvision: {}".format(torchvision.__version__))
    # print("\tCUDA: {}".format(torch.version.cuda))
    # print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    # print("\tNumPy: {}".format(np.__version__))
    # print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    if args.gpus is not None:
        gpus = []
        for num in args.gpus:
            if num in [2, 3]:
                gpus.append(int(num))
    else:
        print("No GPU specified!")
        raise ValueError

    if args.dataset == "LM_PACS_LIGHTNING":
        pass
        dataset = LM_PACS_LIGHTNING()
    else:
        print("Dataset not supported!")
        raise ValueError

    if args.algorithm == "LM_CCVAE_LIGHTNING":
        model = LM_CCVAE_LIGHTNING(num_classes=7,
                                             num_domains=3,
                                             latent_size=args.laten_size,
                                             lamb=args.lamg,
                                             lr=args.lr)
        if args.ckpt_path is not None:
            model.load_state_dict(torch.load(args.ckpt_path))   
    else:
        print("Algorithm not supported!")
        raise ValueError

    trainer = pl.Trainer(gpus=gpus)
    trainer.fit(model, train_dataloader, val_dataloader)
    """
