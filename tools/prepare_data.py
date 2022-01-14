from argparse import ArgumentParser
import numpy as np
import os


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--ratios", type=str, default="811")

    args = parser.parse_args()

    train_ratio = float(args.ratios[0]) * 0.1
    val_ratio = float(args.ratios[1]) * 0.1
    test_ratio = float(args.ratios[2]) * 0.1

    data_dir = "data/" + args.dataset

    print("Configuration:")
    print(f"    train_ratio: {train_ratio}")
    print(f"    val_ratio: {val_ratio}")
    print(f"    test_ratio: {test_ratio}")
    print(f"    data_dir: {data_dir}")
    print("")

    N = 0
    data_dict = {}
    domains = set()
    contents = set()
    for domain in os.listdir(f"{data_dir}"):
        domains.add(domain)
        # print(domain)
        for content in os.listdir(f"{data_dir}/{domain}"):
            # print("    ", content)
            contents.add(content)
            for file in os.listdir(f"{data_dir}/{domain}/{content}"):
                # print("        ", file)
                data_dict[N] = f"{domain}/{content}/{file}"
                N += 1

    print(f"Size of dataset:       {N}")
    inds = np.arange(N)
    np.random.shuffle(inds)
    train_inds = inds[
        0
        :
        int(train_ratio * N)
    ]
    val_inds = inds[
        int(train_ratio * N)
        :
        int(train_ratio * N) + int(val_ratio * N)
    ]
    test_inds = inds[
        int(train_ratio * N) + int(val_ratio * N)
        :
        N
    ]

    print(f"    Size of train set: {len(train_inds)}")
    print(f"    Size of val set:   {len(val_inds)}")
    print(f"    Size of test set:  {len(test_inds)}")

    for domain in domains:
        for content in contents:
            os.makedirs(f"data/{args.dataset}_train/{domain}/{content}")
            os.makedirs(f"data/{args.dataset}_val/{domain}/{content}")
            os.makedirs(f"data/{args.dataset}_test/{domain}/{content}")

    if os.name == "nt":
        copy = "copy"
    else:
        copy = "cp"

    print("Copying train set...")
    for ind in train_inds:
        source = f"data/{args.dataset}/{data_dict[ind]}"
        dest = f"data/{args.dataset}_train/{data_dict[ind]}"
        os.popen(f"{copy} {source} {dest}")

    print("Copying val set...")
    for ind in val_inds:
        source = f"data/{args.dataset}/{data_dict[ind]}"
        dest = f"data/{args.dataset}_val/{data_dict[ind]}"
        os.popen(f"{copy} {source} {dest}")

    print("Copying test set...")
    for ind in test_inds:

        source = f"data/{args.dataset}/{data_dict[ind]}"
        dest = f"data/{args.dataset}_test/{data_dict[ind]}"
        os.popen(f"{copy} {source} {dest}")

    print("Done!")
