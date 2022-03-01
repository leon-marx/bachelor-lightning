import os
import subprocess

model_list = [
    "AAE",
    "CVAE_v3_0-000000",
    "CVAE_v3_0-000100",
    "CVAE_v3_0-010000",
    "MMD_CVAE_0-000000_0-000100_mmd",
    "MMD_CVAE_0-000000_0-010000_mmd",
    "MMD_CVAE_0-000100_0-000100_mmd",
    "MMD_CVAE_0-000100_0-010000_mmd",
    "MMD_CVAE_0-010000_0-000100_mmd",
    "MMD_CVAE_0-010000_0-010000_mmd",
    "MMD_CVAE_0-000000_0-000000_elbo",
    "MMD_CVAE_0-000100_0-000000_elbo",
    "MMD_CVAE_0-010000_0-000000_elbo",
]

for model in model_list:
    print(f"scp -rp -P 49200 tarkus@129.206.118.41:leon/bachelor-lightning/logs/sweep/{model}/version_0/images desktop/bachelor/code/bachelor-lightning/logs/experiment/rmnist_meta1/{model}_images")