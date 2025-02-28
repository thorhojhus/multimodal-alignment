from datasets import DatasetDict, Dataset, concatenate_datasets
import os
import random
import math
import glob
import torchaudio
import numpy as np

def create_melatech_dataset(melatech_data_dir, output_dir=None, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, 
                           chunk_size=1000, seed=42):
    """
    Create a DatasetDict from the melatech-data directory, compatible with AudioChatDataset.
    """
    # Path to the mapping file and audio directory
    mapping_path = os.path.join(melatech_data_dir, "dataset_mapping.txt")
    audio_dir = os.path.join(melatech_data_dir, "audio_files")
    
    # Read the mapping file
    data = []
    with open(mapping_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                audio_relative_path, reader_name, notes = parts
                audio_path = os.path.join(audio_dir, audio_relative_path)
                
                # Only process if audio file exists
                if os.path.exists(audio_path):
                    data.append({
                        "audio_path": audio_path,
                        "reader_name": reader_name,
                        "text": notes
                    })
                else:
                    print(f"Warning: Audio file not found: {audio_path}")
    
    print(f"Loaded {len(data)} examples from mapping file")
    
    # Shuffle the data for random split
    random.seed(seed)
    random.shuffle(data)
    
    # Calculate split sizes
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    test_size = int(total_size * test_ratio)
    
    # Split the data
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    val_data = data[train_size + test_size:]
    
    print(f"Split sizes: train={len(train_data)}, test={len(test_data)}, validation={len(val_data)}")
    
    # Create datasets by processing audio files
    def process_data_chunk(data_list):
        processed_data = []
        
        for item in data_list:
            try:
                # Load audio
                waveform, sample_rate = torchaudio.load(item["audio_path"])
                # Convert to mono if needed
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Add audio data to item
                processed_item = {
                    "audio": {
                        "array": waveform.squeeze().numpy(),
                        "sampling_rate": sample_rate
                    },
                    "text": item["text"],
                    "reader_name": item["reader_name"]
                }
                processed_data.append(processed_item)
            except Exception as e:
                print(f"Error processing {item['audio_path']}: {e}")
        
        return processed_data
    
    # Function to save chunked datasets as arrow files
    def save_chunked_datasets(data_list, prefix, output_dir):
        num_chunks = math.ceil(len(data_list) / chunk_size)
        saved_dirs = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(data_list))
            chunk_data = data_list[start_idx:end_idx]
            
            # Process audio files for this chunk
            processed_chunk = process_data_chunk(chunk_data)
            
            if not processed_chunk:
                continue
                
            # Create the dataset from processed data
            dataset = Dataset.from_list(processed_chunk)
            
            # Save dataset to disk
            dir_path = os.path.join(output_dir, f"{prefix}-{i:02d}")
            dataset.save_to_disk(dir_path)
            
            print(f"Saved {dir_path} with {len(processed_chunk)} examples")
            saved_dirs.append(dir_path)
        
        return saved_dirs
    
    # Save datasets if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_chunked_datasets(train_data, "melatech-train", output_dir)
        save_chunked_datasets(test_data, "melatech-test", output_dir)
        save_chunked_datasets(val_data, "melatech-val", output_dir)
    
    # Process entire dataset for in-memory use
    train_processed = process_data_chunk(train_data)
    test_processed = process_data_chunk(test_data)
    val_processed = process_data_chunk(val_data)
    
    # Create the dataset dictionary
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_processed),
        'test': Dataset.from_list(test_processed),
        'validation': Dataset.from_list(val_processed)
    })
    
    return dataset_dict

def create_load_melatech_module(output_dir):
    """Create a Python module file to load the melatech dataset."""
    module_content = """from datasets import DatasetDict, Dataset, concatenate_datasets
import glob
import os

HOME = os.path.expanduser("~")

melatech_dataset_dict = DatasetDict({
    'train': concatenate_datasets([Dataset.from_disk(f) for f in glob.glob(f"{HOME}/datasets/melatech/melatech-train-*")]),
    'test': concatenate_datasets([Dataset.from_disk(f) for f in glob.glob(f"{HOME}/datasets/melatech/melatech-test-*")]),
    'validation': concatenate_datasets([Dataset.from_disk(f) for f in glob.glob(f"{HOME}/datasets/melatech/melatech-val-*")])
})

if __name__ == "__main__":
    print(f"Train dataset size: {len(melatech_dataset_dict['train'])}")
    print(f"Test dataset size: {len(melatech_dataset_dict['test'])}")
    print(f"Validation dataset size: {len(melatech_dataset_dict['validation'])}")
    breakpoint()
"""
    
    # Create dataset directory if needed
    os.makedirs(os.path.join(output_dir, "dataset"), exist_ok=True)
    
    # Write the module file
    with open(os.path.join(output_dir, "dataset", "load_melatech.py"), 'w') as f:
        f.write(module_content)
    
    print(f"Created load_melatech.py module at {os.path.join(output_dir, 'dataset')}")

if __name__ == "__main__":
    # Example usage
    HOME = os.path.expanduser("~")
    melatech_data_dir = os.path.join(HOME, "work3/shared/datasets/melatech-data")
    output_dir = os.path.join(HOME, "datasets/melatech")
    
    # Create and save the dataset
    dataset_dict = create_melatech_dataset(melatech_data_dir, output_dir)
    
    # Create the load_melatech.py module
    module_dir = os.path.dirname(os.path.dirname(output_dir))
    create_load_melatech_module(module_dir)
    
    print("\nTo use the melatech dataset with AudioChatDataset, add the following code to the __init__ method:")
    print("""elif dataset == "melatech":
    from dataset.load_melatech import melatech_dataset_dict
    self.dataset = melatech_dataset_dict[split.split(".")[0]]""")
    
    breakpoint()