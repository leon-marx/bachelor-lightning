import os
import subprocess
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default=None)
    args = parser.parse_args()

    for folder in os.listdir(args.logdir):
        folder_components = folder.split("_")
        model = folder_components[0]
        data = folder_components[1]
        domains = folder_components[2]
        new_folder = model + "_" + domains + "_" + data

        print(f"Renaming {folder} to {new_folder}")

        bashCommand = f"mv {folder} {new_folder}"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()