from datasets import DatasetDict, Dataset, concatenate_datasets
import glob
import os

HOME = os.path.expanduser("~")
DATASET_PATH = f"{HOME}/datasets/melatech"

train_files = glob.glob(f"{DATASET_PATH}/melatech-train-*/*.arrow")
train_dataset = concatenate_datasets([Dataset.from_file(f) for f in train_files])

val_files = glob.glob(f"{DATASET_PATH}/melatech-val-*/*.arrow")
val_dataset = concatenate_datasets([Dataset.from_file(f) for f in val_files])

test_files = glob.glob(f"{DATASET_PATH}/melatech-test-*/*.arrow")
test_dataset = concatenate_datasets([Dataset.from_file(f) for f in test_files])

melatech_dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset,
})

if __name__ == "__main__":
    # Print some info about the dataset
    print(f"Train set size: {len(melatech_dataset_dict['train'])}")
    print(f"Validation set size: {len(melatech_dataset_dict['validation'])}")
    print(f"Test set size: {len(melatech_dataset_dict['test'])}")
    breakpoint()