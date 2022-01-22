import os
import subprocess

bashCommand = "tensorboard --logdir_spec "
for run in os.listdir("logs"):
    if "long_elu" in run:
        for log in os.listdir(f"logs/{run}/version_0"):
            if "event" in log:
                bashCommand += f"{run}:logs/{run}," 

bashCommand = bashCommand[:-1]
print(bashCommand)

process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()