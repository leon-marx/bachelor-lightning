from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, NDArrayField
from datasets.fast_pacs_balanced import PACSDataset
import numpy as np

# Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
root = "data/"
mode = "train"
domains = ["art_painting", "cartoon", "photo"]
contents = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
my_dataset = PACSDataset(root, mode, sorted(domains), contents)
domain_string = ""
for d in sorted(domains):
    domain_string += d[0]
write_path = f"{root}/fast_paths/PACS_{domain_string}_{mode}"

# Pass a type for each data field
writer = DatasetWriter(write_path, {
    # Tune options to optimize dataset size, throughput at train-time
    "images": RGBImageField(),
    "domains": NDArrayField(dtype=np.int_, shape=(len(domains))),
    "contents": NDArrayField(dtype=np.int_, shape=(len(contents)))
}, num_workers=20)

# Write dataset
writer.from_indexed_dataset(my_dataset)