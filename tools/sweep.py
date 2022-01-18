import copy
import os
import subprocess


def get_combinations(arg_dict_list):
    check = False
    dict_list = []
    for arg_dict in arg_dict_list:
        for key, value in arg_dict.items():
            if isinstance(value, list):
                check = True
                for val in value:
                    d = copy.deepcopy(arg_dict)
                    d[key] = val
                    dict_list.append(d)
    if check:
        return get_combinations(dict_list)
    else:
        return arg_dict_list


if __name__ == "__main__":
    configs = {
        "CVAE": {
            "latent_size": [128, 512, 1024],
            "lamb": [0.01, 1, 100],
        },
        "AE": {
            "latent_size": [128, 512, 1024],
        },
        "PL_AE": {
            "latent_size": [128, 512, 1024],
        },
    }

    for model in configs:
        print(f"Starting loop over {model} configurations.")
        combinations = get_combinations(configs[model])
        for conf in combinations:
            # Default values
            log_dir = f"{model}"
            latent_size =  512
            lamb =  10
            lr =  1e-4

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
                
            bashCommand = f"python -m tools.train\
                --datadir data/variants/PACS_small\
                --batch_size 32\
                --num_workers 20\
                --model {model}\
                --latent_size {latent_size}\
                --lamb {lamb}\
                --lr {lr}\
                --ckpt_path 0\
                --gpus 2,3\
                --output_dir logs/sweep/{log_dir}\
                --max_epochs 100"

            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()