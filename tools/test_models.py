import torch

from datasets.pacs_balanced import BalancedPACSDataModule
from datasets.rotated_mnist_balanced import BalancedRMNISTDataModule
from models.cvae_v3 import CVAE_v3
from models.aae import AAE
from models.aae_v2 import AAE_v2
from models.mmd_cvae import MMD_CVAE

batch_size = 4

pacs_domains = ["art_painting", "cartoon", "photo"]
pacs_contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
pacs_dm = BalancedPACSDataModule(root="data", domains=pacs_domains, contents=pacs_contents,
    batch_size=batch_size, num_workers=0)
pacs_dm.setup()
pacs_batch = next(iter(pacs_dm.train_dataloader()))

rmnist_domains = [0, 15, 30, 45, 60, 75]
rmnist_contents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rmnist_dm = BalancedRMNISTDataModule(root="data", domains=rmnist_domains, contents=rmnist_contents,
    batch_size=batch_size, num_workers=0)
rmnist_dm.setup()
rmnist_batch = next(iter(rmnist_dm.train_dataloader()))

batch_dict = {"PACS": pacs_batch, "RMNIST": rmnist_batch}
domains_dict = {"PACS": pacs_domains, "RMNIST": rmnist_domains}
contents_dict = {"PACS": pacs_contents, "RMNIST": rmnist_contents}

out_channels_dict = [
    {"PACS": [4, 4, 8, 8, 16, 16], "RMNIST": [4, 4, 8]},
    {"PACS": [4, 4, 8, 8, 16, 16], "RMNIST": [4, 4, 8]},
    {"PACS": [4, 4, 8, 8, 16, 16, 32], "RMNIST": [4, 4, 8, 8]},
    {"PACS": [4, 4, 8, 8, 16, 16, 32], "RMNIST": [4, 4, 8, 8]}]
loss_mode_dict = {
    0: "deep_lpips",
    1: "mmd",
    2: "deep_lpips",
    3: "deep_lpips"}

for data in ["PACS", "RMNIST"]:
    batch = batch_dict[data]
    domains = domains_dict[data]
    contents = contents_dict[data]
    num_domains = len(domains)
    num_contents = len(contents)
    latent_size = 128
    feature_size = 8
    mmd_size = 16
    lamb = 0.1
    beta = 0.1
    lr = 1e-4
    depth = 2
    kernel_size = 3
    activation = torch.nn.ELU()
    downsampling = "stride"
    upsampling = "upsample"
    dropout = False
    dropout_rate = 0.2
    batch_norm = True
    no_bn_last = False
    net = "squeeze"
    calibration = "True"
    
    m_cvae = CVAE_v3(data=data, num_domains=num_domains, num_contents=num_contents, 
                latent_size=latent_size, lr=lr, depth=depth, out_channels=out_channels_dict[0][data], 
                kernel_size=kernel_size, activation=activation, downsampling=downsampling, 
                upsampling=upsampling, dropout=dropout, batch_norm=batch_norm, loss_mode=loss_mode_dict[0],
                lamb=lamb, no_bn_last=no_bn_last, initialize=True)
    m_mmd_cvae = MMD_CVAE(data=data, num_domains=num_domains, num_contents=num_contents,
                latent_size=latent_size, lr=lr, depth=depth, 
                out_channels=out_channels_dict[1][data], kernel_size=kernel_size, activation=activation,
                downsampling=downsampling, upsampling=upsampling, dropout=dropout,
                batch_norm=batch_norm, loss_mode=loss_mode_dict[1], lamb=lamb, beta=beta, initialize=True)
    m_aae = AAE(data=data, num_domains=num_domains, num_contents=num_contents,
                latent_size=latent_size, lr=lr, depth=depth, 
                out_channels=out_channels_dict[2][data], kernel_size=kernel_size, activation=activation,
                downsampling=downsampling, upsampling=upsampling, dropout=dropout, loss_mode=loss_mode_dict[2],
                batch_norm=batch_norm, initialize=True)
    m_aae_v2 = AAE_v2(data=data, num_domains=num_domains, num_contents=num_contents,
                latent_size=latent_size, lr=lr, depth=depth, 
                out_channels=out_channels_dict[3][data], kernel_size=kernel_size, activation=activation,
                downsampling=downsampling, upsampling=upsampling, dropout=dropout, loss_mode=loss_mode_dict[3],
                lamb=lamb, net=net, calibration=calibration, batch_norm=batch_norm, initialize=True)

    for i, model in enumerate([m_cvae, m_mmd_cvae, m_aae, m_aae_v2]):
        print(f"model_{i}")
        print(f"    {data}")
        if i <= 1:
            loss = model.training_step(batch, 0)
        else:
            loss = model.training_step(batch, 0, 0)
        print(f"    {loss}")
