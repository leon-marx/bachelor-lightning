from argparse import ArgumentParser
import copy
import os
import subprocess


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
        "CVAE": {
            "latent_size": [128, 512, 1024],
            "lamb": [0.01, 1, 100],
        },
        "AE": {
            "latent_size": [128, 512, 1024],
        },
        "AE_v2": {
            "latent_size": [128, 512, 1024],
        },
        "AE_v3": {
            "out_channels": [
                "128,256,512,512,1024,1024",
                "512,512,512,512,512,512",
                "128,128,256,256,512,512",
                "128,256,512,512,1024,2048"
            ],
            "latent_size": [128, 512, 1024],
            "depth": [1, 2, 3],
            "kernel_size": [3, 5, 7],
            "activation": ["relu", "gelu", "lrelu", "elu"],
            "downsampling": ["stride", "maxpool"],
            "upsampling": ["stride", "upsample"],
            "dropout": [True, False],
            "batch_norm": [True, False],
        },
    }
    ####################
    step = 0
    for model in configs:
        print(f"Starting loop over {model} configurations.")
        combinations = get_combinations(configs[model])
        for conf in combinations:
            if step >= args.start_step:
                print(f"Configuration: {conf}")
                # Default values
                log_dir = f"{model}"
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
                    
                bashCommand = f"python -m tools.train --datadir data/variants/PACS_small --batch_size 8 --num_workers 20 --model {model} --latent_size {latent_size} --lamb {lamb} --lr {lr} --ckpt_path 0 --gpus 2,3 --output_dir logs/sweep/{log_dir} --max_epochs 50 --enable_checkpointing False --depth {depth} --out_channels {out_channels} --kernel_size {kernel_size} --activation {activation} --downsampling {downsampling} --upsampling {upsampling} --dropout {dropout} --batch_norm {batch_norm}"
                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                print(f"Completed step {step}!")
            step += 1