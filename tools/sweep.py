import copy
import os
import subprocess


if __name__ == "__main__":
    model_list = ["CVAE", "AE", "PL_AE"]

    cvae_args = {
            "latent_size": [128, 512, 1024],
            "lamb": [0.01, 1, 100],
    }
    ae_args = {
            "latent_size": [128, 1024],
    }

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
            return dict_list

    print(get_combinations([cvae_args]))
    print(get_combinations([ae_args]))
        
    for model in model_list:
        bashCommand = "python -m tools.train\
            --datadir data/variants/PACS_small\
            --batch_size 32\
            --num_workers 20\
            --model AE\
            --latent_size 512\
            --lamb 10\
            --lr 1e-03\
            --ckpt_path 0\
            --gpus 2\
            --output_dir logs/ae"

        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()