import os
import subprocess

# bashCommand = "python -m tools.train --batch_size 2 --num_workers 4 --model AAE --latent_size 128 --depth 1 --kernel_size 3 --activation selu --downsampling stride --upsampling upsample --lr 1e-04 --ckpt_path 0 --gpus 0 --output_dir logs/test --out_channels 16,16,32,32,64,64,128 --loss_mode deep"
# bashCommand = "python -m debugging.debug"
bashCommand = "python -m debugging.debug_sweep_classifier --data RMNIST --log_name trash --models CNN,ERM --test_mode --gpus 0 --max_epochs 100 --batch_size 1"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()