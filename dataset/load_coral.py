from datasets import DatasetDict, Dataset, concatenate_datasets
import glob
import os

HOME = os.path.expanduser("~")

coral_dataset_dict = DatasetDict({
    'train': concatenate_datasets([Dataset.from_file(f) for f in glob.glob(f"{HOME}/datasets/coral/coral-train-*.arrow")]),
    'test': concatenate_datasets([Dataset.from_file(f) for f in glob.glob(f"{HOME}/datasets/coral/coral-test-*.arrow")]),
    'validation': concatenate_datasets([Dataset.from_file(f) for f in glob.glob(f"{HOME}/datasets/coral/coral-val-*.arrow")])
})

if __name__ == "__main__":
    breakpoint()