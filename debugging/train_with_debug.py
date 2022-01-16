import os
import subprocess

bashCommand = "python -m tools.train --batch_size 2 --num_workers 4 --model AE --latent_size 512 --lamb 10 --lr 1e-03 --ckpt_path 0 --gpus 0 --output_dir logs/test"

process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()