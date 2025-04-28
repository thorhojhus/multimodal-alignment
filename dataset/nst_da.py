from datasets import DatasetDict, Dataset, concatenate_datasets
import glob
import os

HOME = os.path.expanduser("~")


path = f"{HOME}/datasets/nst-da/data/"

nst_dataset_dict = DatasetDict({
    'train': concatenate_datasets([Dataset.from_file(f) for f in glob.glob(path + "*train*")]),
    'test': concatenate_datasets([Dataset.from_file(f) for f in glob.glob(path + "*test*")]),
})

if __name__ == "__main__":
    breakpoint()