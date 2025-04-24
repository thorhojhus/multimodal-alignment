import os
import pandas as pd
from phonemizer import phonemize
import truecase

# Directories to process
directories = [
    'test_clean',
    'test_other',
    'train_clean_100',
    'train_clean_360',
    'train_other_500',
    'validation_clean',
    'validation_other'
]

# Base path
base_path = '~/shared/datasets/librispeech_files/'
base_path = os.path.expanduser(base_path)

# Process each directory
for directory in directories:
    dir_path = os.path.join(base_path, directory)
    input_file = os.path.join(dir_path, 'data.csv')
    output_file = os.path.join(dir_path, 'data_phonemized.csv')

    if not os.path.exists(output_file):
        print(f"Processing {input_file}...")
        
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Apply truecasing
        df['text'] = df['text'].apply(lambda text: truecase.get_true_case(text))
        
        try:
            # Apply phonemization
            df['text'] = df['text'].apply(lambda text: phonemize(
                text,
                language='en-us',
                backend='espeak',
                words_mismatch="ignore",
                strip=True,
                preserve_punctuation=True,
                with_stress=False,
                logger=None
            ))
        except RuntimeError as e:
            print(f"Error phonemizing {input_file}: {e}")
            continue
        
        # Save the result
        df.to_csv(output_file, index=False)
        print(f"Created {output_file}")

print("All directories processed successfully!")