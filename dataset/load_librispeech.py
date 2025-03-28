from datasets import DatasetDict, Dataset, concatenate_datasets
import glob
import os

HOME = os.path.expanduser("~")

librispeech_dataset_dict = DatasetDict({
    'train.clean.100': concatenate_datasets([Dataset.from_file(f) for f in glob.glob(f"{HOME}/datasets/librispeech/librispeech_asr-train.clean.100-*.arrow")]),
    'train.clean.360': concatenate_datasets([Dataset.from_file(f) for f in glob.glob(f"{HOME}/datasets/librispeech/librispeech_asr-train.clean.360-*.arrow")]),
    'train.other.500': concatenate_datasets([Dataset.from_file(f) for f in glob.glob(f"{HOME}/datasets/librispeech/librispeech_asr-train.other.500-*.arrow")]),
    'test.clean': Dataset.from_file(f"{HOME}/datasets/librispeech/librispeech_asr-test.clean.arrow"),
    'test.other': Dataset.from_file(f"{HOME}/datasets/librispeech/librispeech_asr-test.other.arrow"),
    'validation.clean': Dataset.from_file(f"{HOME}/datasets/librispeech/librispeech_asr-validation.clean.arrow"),
    'validation.other': Dataset.from_file(f"{HOME}/datasets/librispeech/librispeech_asr-validation.other.arrow")
})

librispeech_dataset_dict['train'] = concatenate_datasets([
    librispeech_dataset_dict['train.clean.100'],
    librispeech_dataset_dict['train.clean.360'],
    librispeech_dataset_dict['train.other.500']
])

if __name__ == "__main__":
    breakpoint()