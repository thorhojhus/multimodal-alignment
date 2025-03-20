import os
from datasets import load_dataset

os.makedirs(os.path.expanduser("~/datasets/melatech"), exist_ok=True)

print("Loading dataset...")
dataset = load_dataset(
    "audiofolder", 
    data_dir="~/datasets/melatech-data",
    num_proc=8,
)

print("Splitting dataset into train/test...")
train_test_datasets = dataset["train"].train_test_split(test_size=0.1, seed=42)

def process_audio(example):
    example["text"] = example.pop("additional_feature")
    return example

print("Processing train set...")
train_dataset = train_test_datasets["train"].map(
    process_audio,
    num_proc=8,
)

print("Processing test set...")
test_dataset = train_test_datasets["test"].map(
    process_audio,
    num_proc=8,
)

print("Saving train set to arrow files...")
train_path = os.path.expanduser("~/datasets/melatech/train")
train_dataset.save_to_disk(train_path)

print("Saving test set to arrow files...")
test_path = os.path.expanduser("~/datasets/melatech/test")
test_dataset.save_to_disk(test_path)

print(f"Dataset processed and saved successfully.")
print(f"Train set: {len(train_dataset)} samples, Test set: {len(test_dataset)} samples")
print(f"Files saved to: {os.path.expanduser('~/datasets/melatech/')}")