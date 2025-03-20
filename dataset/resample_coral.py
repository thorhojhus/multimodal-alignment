import os
import glob
import tqdm
import numpy as np
from datasets import Dataset
from subprocess import run, CalledProcessError
import tempfile
import soundfile as sf

# Constants
SAMPLE_RATE = 16000

def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def process_example(example):
    """Process a single example, resampling audio to 16kHz using ffmpeg"""
    # Make a copy to avoid modifying the original
    result = dict(example)
    
    if 'audio' in example and isinstance(example['audio'], dict):
        audio_array = example['audio']['array']
        sample_rate = example['audio']['sampling_rate']
        
        if sample_rate != SAMPLE_RATE:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                sf.write(temp_file.name, audio_array, sample_rate)
                resampled = load_audio(temp_file.name, SAMPLE_RATE)
                result['audio'] = {
                    'array': resampled,
                    'sampling_rate': SAMPLE_RATE
                }
        
    return result

def process_split(split_name, input_dir, output_dir, num_proc=8):
    """Process all files for a specific split"""
    # Get all input files for this split
    input_files = glob.glob(f"{input_dir}/coral-{split_name}-*.arrow")
    total_files = len(input_files)
    print(f"Found {total_files} files for {split_name} split")
    
    # Process each file with a progress bar
    for i, input_file in enumerate(tqdm.tqdm(input_files, desc=f"Processing {split_name}", unit="file")):
        # Get the filename to preserve structure
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        
        # Load dataset
        dataset = Dataset.from_file(input_file)
        
        # Resample audio using parallel processing with dataset.map
        resampled_dataset = dataset.map(
            process_example,
            num_proc=num_proc,  # Use parallel processing
            desc=f"Resampling file {i+1}/{total_files}"
        )
        
        # Save as arrow file directly
        with open(output_file, 'wb') as f:
            f.write(resampled_dataset.to_arrow_bytes())
            
        tqdm.tqdm.write(f"Completed file {i+1}/{total_files}: {filename} ({len(dataset)} examples)")
    
    print(f"Completed all {total_files} files for {split_name} split")

def main():
    HOME = os.path.expanduser("~")
    
    # Define input and output paths
    coral_path = f"{HOME}/datasets/coral"
    resampled_path = f"{HOME}/datasets/coral-resampled"
    
    # Create output directory
    os.makedirs(resampled_path, exist_ok=True)
    
    total_splits = 3
    for i, split in enumerate(['train', 'test', 'validation']):
        print(f"[{i+1}/{total_splits}] Processing {split} split...")
        process_split(split, coral_path, resampled_path, num_proc=12)
    
    print("Resampling complete!")

if __name__ == "__main__":
    main()