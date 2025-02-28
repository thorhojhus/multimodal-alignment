from datasets import DatasetDict, Dataset, concatenate_datasets
import glob
import os

HOME = os.path.expanduser("~")

melatech_dataset_dict = DatasetDict({
    'train': concatenate_datasets([Dataset.from_file(f) for f in glob.glob(f"{HOME}/datasets/melatech/melatech-train-*.arrow")]),
    'test': concatenate_datasets([Dataset.from_file(f) for f in glob.glob(f"{HOME}/datasets/melatech/melatech-test-*.arrow")]),
    'validation': concatenate_datasets([Dataset.from_file(f) for f in glob.glob(f"{HOME}/datasets/melatech/melatech-val-*.arrow")])
})


if __name__ == "__main__":
    melatech_dataset_dict
    breakpoint()