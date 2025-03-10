from datasets import DatasetDict, Dataset
import os
import pandas as pd
import soundfile as sf
import numpy as np

class CommonVoiceArDataset(Dataset):    
    def __init__(self, df, clips_dir):
        self.df = df
        self.clips_dir = clips_dir
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.clips_dir, row['path'])
        
        # Load audio file on-demand
        try:
            array, sampling_rate = sf.read(audio_path)
            
            # Ensure float32
            if array.dtype != np.float32:
                array = array.astype(np.float32)
            
            # Ensure mono audio
            if len(array.shape) > 1:
                array = array.mean(axis=1)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            array = np.zeros(16000, dtype=np.float32)
            sampling_rate = 16000
        
        return {
            "audio": {
                "array": array,
                "sampling_rate": sampling_rate
            },
            "text": row['sentence']
        }

def load_common_voice_ar(data_dir):

    split_files = {
        'train': 'train.tsv',
        'validation': 'dev.tsv',
        'test': 'test.tsv'
    }
    
    dataset_dict = {}
    
    for split, file_name in split_files.items():
        tsv_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(tsv_path, sep='\t')
        
        dataset_dict[split] = CommonVoiceArDataset(df, os.path.join(data_dir, "clips"))
    
    return DatasetDict(dataset_dict)

common_voice_ar_dataset_dict = load_common_voice_ar("/workspace/arabic-data/cv-corpus-20.0-2024-12-06/ar")