from datasets import DatasetDict, interleave_datasets, load_dataset
import os

HOME = os.path.expanduser("~")

# List of languages
languages = ["german", "dutch", "french", "italian", "spanish", "polish", "portuguese"]

# Function to process examples - keep only audio and rename transcript to text
def process_example(example):
    return {
        "audio": example["audio"],
        "text": example["transcript"]
    }

# Load and process streaming datasets for training
train_datasets = [
    load_dataset("facebook/multilingual_librispeech", lang, split="train")
    .map(process_example)
    .remove_columns(["original_path", "begin_time", "end_time", "audio_duration", "transcript", "speaker_id", " chapter_id", "file", "id"])
    .with_format("torch")
    for lang in languages
]

# Load and process streaming datasets for testing
test_datasets = [
    load_dataset("facebook/multilingual_librispeech", lang, split="test")
    .map(process_example)
    .remove_columns(["original_path", "begin_time", "end_time", "audio_duration", "transcript", "speaker_id", " chapter_id", "file", "id"])
    .with_format("torch")
    for lang in languages
]

# Interleave datasets for better streaming performance
combined_train = interleave_datasets(train_datasets).with_format("torch")
combined_test = interleave_datasets(test_datasets).with_format("torch")

# Create DatasetDict with both train and test splits
multilingual_librispeech_dataset_dict = DatasetDict({
    "train": combined_train,
    "test": combined_test
})