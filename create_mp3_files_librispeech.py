import os
import csv
import tempfile
from functools import partial
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import soundfile as sf
from pydub import AudioSegment
from dataset.load_librispeech import librispeech_dataset_dict

# Define base directory
base_dir = os.path.expanduser("~/shared/datasets/librispeech_files")
os.makedirs(base_dir, exist_ok=True)

# Define splits
splits = [
    #"train.clean.100",
    #"train.clean.360",
    #"train.other.500",
    "test.clean",
    "test.other",
    "validation.clean",
    "validation.other"
]

# Load LibriSpeech dataset
print("Loading LibriSpeech dataset...")
librispeech = librispeech_dataset_dict

def process_example(example, full_dir):
    """
    Processes a single example:
      - Converts the audio to MP3 at 16kHz.
      - Saves the file in full_dir.
      - Returns a tuple (filename, text).
    """
    audio_filename = f"{example['id']}.mp3"
    audio_path = os.path.join(full_dir, audio_filename)

    # Get audio data and sample rate
    audio_array = example['audio']['array']
    sample_rate = example['audio']['sampling_rate']

    # Convert to MP3 at 16kHz
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
        # Save as temporary WAV file
        sf.write(temp_wav.name, audio_array, sample_rate)

        # Load and adjust audio if needed
        audio = AudioSegment.from_wav(temp_wav.name)
        if sample_rate != 16000:
            audio = audio.set_frame_rate(16000)
        audio.export(audio_path, format="mp3")
    
    return (audio_filename, example['text'])

# Process each split
for split in splits:
    print(f"Processing {split}...")

    # Create directory for this split by replacing dots with underscores
    splitdir = split.replace(".", "_")
    full_dir = os.path.join(base_dir, splitdir)
    os.makedirs(full_dir, exist_ok=True)

    # Prepare CSV file path
    csv_path = os.path.join(full_dir, "data.csv")
    examples = librispeech[split]

    # Process examples in parallel using 8 cores
    with ProcessPoolExecutor(max_workers=8) as executor:
        process_func = partial(process_example, full_dir=full_dir)
        results = list(tqdm(executor.map(process_func, examples), total=len(examples)))

    # Write results to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['file', 'text'])
        for row in results:
            csv_writer.writerow(row)

    print(f"Completed processing {split}")

print("All processing complete!")
