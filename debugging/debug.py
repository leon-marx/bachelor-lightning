# import torch
# from models.cvae_v3 import CVAE_v3

if __name__ == "__main__":

    ckpt_path = "logs/rmnist_loo_vaes/01234/version_0/checkpoints/epoch=*.cpt"
    hparam_path = ""
    for dir in ckpt_path.split("/")[:-2]:
        hparam_path += dir + "/"
    hparam_path += "hparams.yaml"
    print(hparam_path)

    # def get_activation(act_string):
    #     if "ReLU" in act_string:
    #         activation = torch.nn.ReLU()
    #     if "SELU" in act_string:
    #         activation = torch.nn.SELU()
    #     if "ELU" in act_string:
    #         activation = torch.nn.ELU()
    #     return activation

    # out_channels = []
    # with open("logs/hparams.yaml", "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.strip()
    #         line = line.split(" ")
    #         print(line)
    #         if "data" in line[0]:
    #             data = str(line[1])
    #         if "num_domains" in line[0]:
    #             num_domains = int(line[1])
    #         if "num_contents" in line[0]:
    #             num_contents = int(line[1])
    #         if "latent_size" in line[0]:
    #             latent_size = int(line[1])
    #         if "depth" in line[0]:
    #             depth = int(line[1])
    #         if "kernel_size" in line[0]:
    #             kernel_size = int(line[1])
    #         if "activation" in line[0]:
    #             activation = get_activation(str(line[1]))
    #         if "downsampling" in line[0]:
    #             downsampling = str(line[1])
    #         if "upsampling" in line[0]:
    #             upsampling = str(line[1])
    #         if "dropout" in line[0]:
    #             dropout = line[1] == "true"
    #         if "batch_norm" in line[0]:
    #             batch_norm = line[1] == "true"
    #         if "loss_mode" in line[0]:
    #             loss_mode = str(line[1])
    #         if "lamb" in line[0]:
    #             lamb = float(line[1])
    #         if "no_bn_last" in line[0]:
    #             no_bn_last = line[1] == "true"
    #         if "-" in line[0]:
    #             try:
    #                 val = int(line[1])
    #                 out_channels.append(val)
    #             except ValueError:
    #                 print("ValueError:", line[1])
    
    # model = CVAE_v3(data="PACS", num_domains=num_domains, num_contents=num_contents,
    # latent_size=latent_size, lr=1e-4, depth=depth, 
    # out_channels=out_channels, kernel_size=kernel_size, activation=activation,
    # downsampling=downsampling, upsampling=upsampling, dropout=dropout,
    # batch_norm=batch_norm, loss_mode=loss_mode, lamb=lamb, no_bn_last=no_bn_last)

    # print(data)
    # print(num_domains)
    # print(num_contents)
    # print(latent_size)
    # print(depth)
    # print(out_channels)
    # print(kernel_size)
    # print(activation)
    # print(downsampling)
    # print(upsampling)
    # print(dropout)
    # print(batch_norm)
    # print(loss_mode)
    # print(lamb)
    # print(no_bn_last)

    # print(model)
    # print("done")