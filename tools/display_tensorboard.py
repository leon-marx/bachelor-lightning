import os
import subprocess

bashCommand = "tensorboard --logdir_spec "
for run in os.listdir("logs/sweep"):
    for log in os.listdir(f"logs/sweep/{run}/version_0"):
        if "event" in log:
            bashCommand += f"{run}:logs/sweep/{run}," 

bashCommand = bashCommand[:-1]
print(bashCommand)

process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()