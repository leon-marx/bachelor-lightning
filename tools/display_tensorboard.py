import os
import subprocess

bashCommand = "tensorboard"
for run in os.listdir("logs/sweep"):
    for log in os.listdir(f"logs/sweep/{run}/version_0"):
        if "events" in log:
            bashCommand += f" --logdir={run}:/logs/sweep/{run}/version_0/{log}"

process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()