import os
import subprocess

bashCommand = "python -m tools.train --gpus 0"

process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()