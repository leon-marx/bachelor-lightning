from argparse import ArgumentParser
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, NDArrayField
from datasets.fast_pacs_balanced import PACSDataset
import numpy as np


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--domains", type=str, default="PAC")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    # Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
    root = args.root
    mode = args.mode
    domain_dict = {
        "A": "art_painting",
        "C": "cartoon",
        "P": "photo",
        "S": "sketch",
    }
    domains = sorted([domain_dict[k] for k in args.domains])
    contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
    my_dataset = PACSDataset(root, mode, sorted(domains), contents)
    domain_string = ""
    for d in sorted(domains):
        domain_string += d[0]
    write_path = f"{root}/fast_paths/PACS_{domain_string}_{mode}"

    # Pass a type for each data field
    writer = DatasetWriter(write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        "images": NDArrayField(dtype=np.dtype("int32"), shape=(3, 3, 227, 227)),
        "domains": NDArrayField(dtype=np.dtype("int64"), shape=(3, len(domains))),
        "contents": NDArrayField(dtype=np.dtype("int64"), shape=(3, len(contents)))
    }, num_workers=20)

    # Write dataset
    writer.from_indexed_dataset(my_dataset)