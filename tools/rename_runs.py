import os
import subprocess
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default=None)
    args = parser.parse_args()

    os.chdir(f"logs/{args.logdir}")

    for folder in os.listdir():
        folder_components = folder.split("_")
        model = folder_components[0]
        data = folder_components[2]
        domains = folder_components[1]
        # new_folder = model + "_" + domains + "_" + data
        new_folder = model + "_" + data + "_" + domains

        print(f"Renaming {folder} to {new_folder}")

        bashCommand = f"mv {folder} {new_folder}"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()