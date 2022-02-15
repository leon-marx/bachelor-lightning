import matplotlib.pyplot as plt
import torch
import torchvision
from tqdm import tqdm
import pytorch_lightning as pl
import os

from models.cvae_v3 import CVAE_v3
from callbacks.logger import Logger
from datasets.rotated_mnist import RMNISTDataModule

def check_overfit(trainer, pl_module):
    with torch.no_grad():
        gpus = "0,"
        output_dir = "logs/trash"
        max_epochs = 25
        disable_checkpointing = True
        log_every_n_steps = 5
        data = "RMNIST"
        num_domains = 6
        num_contents = 10
        latent_size = 128
        lr = 1e-4
        out_channels = [128, 256, 512, 512, 1024, 1024]
        depth = 1
        kernel_size = 3
        activation = torch.nn.ELU()
        downsampling = "stride"
        upsampling = "upsample"
        dropout = False
        batch_norm = True
        loss_mode = "elbo"
        lamb = 0.1
        no_bn_last = True
        batch_size = 8

        domains = [0, 15, 30, 45, 60, 75]
        contents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        domain_dict = {domain: torch.LongTensor([i]) for i, domain in enumerate(domains)}
        content_dict = {content: torch.LongTensor([i]) for i, content in enumerate(contents)}

        pl_module.eval()
        bs = 4
        codes = torch.randn(size=(bs, pl_module.latent_size)).to(pl_module.device)
        generated_dict = {
            domain: {content: torch.zeros(size=(bs, 1, 28, 28)) for content in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]} for domain in [0, 15, 30, 45, 60, 75]
        }
        best_reconstruction_dict = {
            domain: {content: torch.zeros(size=(bs, 1, 28, 28)) for content in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]} for domain in [0, 15, 30, 45, 60, 75]
        }
        best_original_dict = {
            domain: {content: torch.zeros(size=(bs, 1, 28, 28)) for content in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]} for domain in [0, 15, 30, 45, 60, 75]
        }
        reconstruction_score_dict = {
            domain: {content: torch.ones(size=(bs,)) * 1000000 for content in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]} for domain in [0, 15, 30, 45, 60, 75]
        }
        original_score_dict = {
            domain: {content: torch.ones(size=(bs,)) * 1000000 for content in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]} for domain in [0, 15, 30, 45, 60, 75]
        }
        # generating all possible domain-content combinations (times 4)
        for domain_name in [0, 15, 30, 45, 60, 75]:
            for content_name in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                doms = torch.nn.functional.one_hot(domain_dict[domain_name], num_classes=len(domains)).repeat(codes.shape[0], 1).to(pl_module.device)
                conts = torch.nn.functional.one_hot(content_dict[content_name], num_classes=len(contents)).repeat(codes.shape[0], 1).to(pl_module.device)
                generated_images = pl_module.generate(codes, doms, conts)
                generated_dict[domain_name][content_name] = generated_images

        # comparing whole training set with generated images
        for batch in tqdm(log_dm.train_dataloader()):
            images = batch[0].to(pl_module.device)
            doms = batch[1].to(pl_module.device)
            conts = batch[2].to(pl_module.device)
            reconstructions = pl_module.reconstruct(images, doms, conts)
            doms = torch.argmax(doms, dim=1)
            conts = torch.argmax(conts, dim=1)
            for i in range(batch_size):
                img = images[i]
                dom = int(doms[i].item() * 15)
                cont = int(conts[i].item())
                rec = reconstructions[i]
                for j in range(bs):
                    rec_score = torch.nn.functional.mse_loss(generated_dict[dom][cont][j], rec)
                    if rec_score < reconstruction_score_dict[dom][cont][j]:
                        reconstruction_score_dict[dom][cont][j] = rec_score
                        best_reconstruction_dict[dom][cont][j] = rec
                    orig_score = torch.nn.functional.mse_loss(generated_dict[dom][cont][j], img)
                    if orig_score < original_score_dict[dom][cont][j]:
                        original_score_dict[dom][cont][j] = orig_score
                        best_original_dict[dom][cont][j] = img
        
        # making grids for each domain-content pair and all 4 generated images
        for dom in [0, 15, 30, 45, 60, 75]:
            for cont in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                canvas = torch.stack((generated_dict[dom][cont].cpu(), best_reconstruction_dict[dom][cont].cpu(), best_original_dict[dom][cont].cpu()), dim=1).view(-1, 1, 28, 28)
                gen_grid = torchvision.utils.make_grid(canvas)
                torchvision.utils.save_image(gen_grid, f"{output_dir}/version_{trainer.logger.version}/images/generated_{dom}_{cont}_comparison.png")
                trainer.logger.experiment.add_image(f"generated_{dom}_{cont}_comparison.png", gen_grid)

        pl_module.train()

if __name__ == "__main__":
    gpus = "0,"
    output_dir = "logs/trash"
    max_epochs = 25
    disable_checkpointing = True
    log_every_n_steps = 5
    data = "RMNIST"
    num_domains = 6
    num_contents = 10
    latent_size = 128
    lr = 1e-4
    out_channels = [128, 256, 512, 512, 1024, 1024]
    depth = 1
    kernel_size = 3
    activation = torch.nn.ELU()
    downsampling = "stride"
    upsampling = "upsample"
    dropout = False
    batch_norm = True
    loss_mode = "elbo"
    lamb = 0.1
    no_bn_last = True
    batch_size = 8

    domains = [0, 15, 30, 45, 60, 75]
    contents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    log_dm = RMNISTDataModule(root="data", domains=domains, contents=contents,
                        batch_size=batch_size, num_workers=4, shuffle_all=True)
    log_dm.setup()
    # train_batch = next(iter(log_dm.train_dataloader()))
    # val_batch = next(iter(log_dm.val_dataloader()))
    train_batch = torch.zeros(size=(3,batch_size,1,28,28))
    val_batch = torch.zeros(size=(3,batch_size,1,28,28))


    callbacks = [
        Logger(output_dir, log_dm, train_batch, val_batch, domains, contents, images_on_val=True),
        pl.callbacks.ModelCheckpoint(monitor="val_loss", save_last=True),
    ]

    trainer = pl.Trainer(
        gpus=gpus,
        # strategy="dp",
        precision=16,
        default_root_dir=output_dir,
        logger=pl.loggers.TensorBoardLogger(save_dir=os.getcwd(),
                                            name=output_dir),
        callbacks=callbacks,
        # gradient_clip_val=1.0,
        # gradient_clip_algorithm="norm",
        max_epochs=max_epochs,
        # enable_checkpointing= not disable_checkpointing,
        # log_every_n_steps=log_every_n_steps
    )
    model = CVAE_v3(data=data, num_domains=num_domains, num_contents=num_contents, 
                        latent_size=latent_size, lr=lr, depth=depth, out_channels=out_channels, 
                        kernel_size=kernel_size, activation=activation, downsampling=downsampling, 
                        upsampling=upsampling, dropout=dropout, batch_norm=batch_norm, loss_mode=loss_mode,
                        lamb=lamb, no_bn_last=no_bn_last, initialize=True)
    check_overfit(trainer, model)
    print("done")
