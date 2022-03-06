import os
import torch

if __name__ == "__main__":
    root = "data/variants/RMNIST_augmented"
    # dom_groups = ["01234", "01235", "01245", "01345", "02345", "12345"]
    dom_groups = ["01234", "01235"]
    for dg in dom_groups:
        print(f"{dg}")
        for dom_dir in os.listdir(f"{root}/RMNIST_train_{dg}"):
            print(f"  {dom_dir}")
            for cont_dir in os.listdir(f"{root}/RMNIST_train_{dg}/{dom_dir}"):
                print(f"    {cont_dir}")
                for datafile in os.listdir(f"{root}/RMNIST_train_{dg}/{dom_dir}/{cont_dir}"):
                    data = torch.load(f"{root}/RMNIST_train_{dg}/{dom_dir}/{cont_dir}/{datafile}")
                    data = data.view(-1, 1, 28, 28)
                    torch.save(data, f"{root}/RMNIST_train_{dg}/{dom_dir}/{cont_dir}/{datafile}")