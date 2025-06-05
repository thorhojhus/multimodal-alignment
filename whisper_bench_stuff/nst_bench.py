import os
import torch
import torchaudio
import transformers
from jiwer import wer, cer
from tqdm import tqdm
import re
import unicodedata
from torch.utils.data import DataLoader, Dataset
from dataset.load_coral import coral_dataset_dict
from dataset.nst_da import nst_dataset_dict
from dataset.audio_dataset import filter_difficult_examples
from datasets import Audio
import pandas as pd

# Set various CUDA and cudnn settings
torch.set_float32_matmul_precision('medium')
torch.cuda.set_per_process_memory_fraction(0.5, device=torch.cuda.current_device())


def normalize_numbers_danish(text):
    """
    Convert Danish number words to digits, handling complex number phrases.
    """
    # Define number mappings
    basic_numbers = {
        "nul": 0, "en": 1, "et": 1, "to": 2, "tre": 3, "fire": 4, "fem": 5,
        "seks": 6, "syv": 7, "otte": 8, "ni": 9, "ti": 10, "elleve": 11,
        "tolv": 12, "tretten": 13, "fjorten": 14, "femten": 15, "seksten": 16,
        "sytten": 17, "atten": 18, "nitten": 19
    }
    
    tens = {
        "tyve": 20, "tredive": 30, "fyrre": 40, "halvtreds": 50,
        "tres": 60, "halvfjerds": 70, "firs": 80, "halvfems": 90
    }
    
    multipliers = {
        "hundrede": 100, "tusind": 1000, "million": 1000000, "millioner": 1000000,
        "milliard": 1000000000, "milliarder": 1000000000
    }
    
    def parse_number_sequence(words, start_idx=0):
        """Parse a complete number sequence and return (value, end_index)"""
        if start_idx >= len(words):
            return 0, start_idx
            
        current_value = 0
        accumulator = 0
        i = start_idx
        
        while i < len(words):
            word = words[i]
            
            # Handle compound numbers (e.g., "treogtyve")
            if "og" in word and word != "og":
                parts = word.split("og")
                if len(parts) == 2:
                    if parts[0] in basic_numbers and parts[1] in tens:
                        accumulator = basic_numbers[parts[0]] + tens[parts[1]]
                        i += 1
                        continue
                    elif parts[0] in basic_numbers and parts[1] in basic_numbers:
                        # Handle cases like "toogtyve" (22)
                        accumulator = basic_numbers[parts[0]] + basic_numbers[parts[1]]
                        i += 1
                        continue
            
            # Handle basic numbers
            if word in basic_numbers:
                if accumulator == 0:
                    accumulator = basic_numbers[word]
                else:
                    # If we already have a number, this starts a new one
                    break
                i += 1
                
            # Handle tens
            elif word in tens:
                if accumulator == 0:
                    accumulator = tens[word]
                else:
                    # If we already have a number, this starts a new one
                    break
                i += 1
                
            # Handle multipliers
            elif word in multipliers:
                if accumulator == 0:
                    accumulator = 1  # Default multiplier
                current_value += accumulator * multipliers[word]
                accumulator = 0
                i += 1
                
            # Handle "og" (and) - continue parsing
            elif word == "og":
                i += 1
                
            else:
                # Not a number word, stop parsing
                break
        
        # Add any remaining accumulator
        current_value += accumulator
        
        return current_value, i
    
    # Process the text
    words = text.split()
    result = []
    i = 0
    
    while i < len(words):
        # Try to parse a number sequence
        value, next_i = parse_number_sequence(words, i)
        
        if next_i > i:
            # We parsed a number
            result.append(str(value))
            i = next_i
        else:
            # Not a number, keep the word as is
            result.append(words[i])
            i += 1
    
    return " ".join(result)

def normalize_text(text):
    """
    Normalize text to lowercase a-z, æ, ø, å, 0-9, and spaces only.
    Convert number words to digits and remove all other characters.
    """
    # Convert to lowercase
    text = text.lower()
    text = text.strip()
    
    danish_chars = {'æ': 'æ', 'ø': 'ø', 'å': 'å'}
    normalized = unicodedata.normalize('NFD', text)
    for dk_char, replacement in danish_chars.items():
        normalized = normalized.replace(dk_char, replacement)
    
    # Remove accents/diacritics
    text = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    
    # Convert number words to digits
    text = normalize_numbers_danish(text)
    
    # Keep only a-z, æ, ø, å, 0-9, and spaces
    text = re.sub(r'[^a-zæøå0-9 ]', '', text)
    
    return text
# Use OpenAI's Whisper model
model_name = "openai/whisper-large-v3"
processor = transformers.WhisperProcessor.from_pretrained(model_name)

class NSTDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio_array = sample["audio"]["array"] 
        return audio_array, sample["text"].lower()

def collate_fn(batch):
    audios, refs = zip(*batch)
    inputs = processor(
        audio=list(audios),
        return_tensors="pt", 
        sampling_rate=16000, 
        padding=True,
        return_attention_mask=True
    )
    return inputs, list(refs)

test_dataset = nst_dataset_dict["test"]
print(test_dataset)
test_dataset = test_dataset.filter(filter_difficult_examples)

print(test_dataset)
test_dataset = NSTDatasetWrapper(test_dataset)

loader = DataLoader(
    test_dataset,
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=8,
    collate_fn=collate_fn,
)

device = torch.device("cuda")

model = transformers.WhisperForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
    device_map="auto",
)
model = model.to(device)

# Enable static cache and compile
model.generation_config.cache_implementation = "static"
model.generation_config.max_new_tokens = 256
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
model.eval()

# Define forced decoder IDs
forced_decoder_ids = processor.get_decoder_prompt_ids(language="danish", task="transcribe")

print("Running warmup...")
# Create dummy audio for warmup (1 second of silence at 16kHz)
dummy_audio = torch.zeros(16000).numpy()

for _ in range(2):
    with torch.inference_mode():
        dummy_inputs = processor(
            dummy_audio, 
            return_tensors="pt", 
            sampling_rate=16000,
            return_attention_mask=True
        )
        dummy_input_features = dummy_inputs.input_features.to(device, torch.float16)
        dummy_attention_mask = dummy_inputs.attention_mask.to(device)
        
        _ = model.generate(
            dummy_input_features,
            attention_mask=dummy_attention_mask,
            forced_decoder_ids=forced_decoder_ids,
            min_new_tokens=256,
            max_new_tokens=256
        )
forced_decoder_ids = processor.get_decoder_prompt_ids(language="danish", task="transcribe")

total_wer = 0.0
total_cer = 0.0
num_batches = 0

all_preds = []
all_refs = []

for inputs, batch_refs in tqdm(loader, desc="Processing batches"):
    input_features = inputs["input_features"].to(device, torch.float16, non_blocking=True)
    attention_mask = inputs["attention_mask"].to(device, non_blocking=True)

    # Use autocast in bfloat16 mode
    with torch.inference_mode():
        generated = model.generate(
            input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_decoder_ids,
            num_beams=1,
            length_penalty=1.0,
            use_cache=True
        )
    
    predictions = processor.batch_decode(generated, skip_special_tokens=True)
    
    # Normalize predictions and references
    normalized_predictions = [normalize_text(pred) for pred in predictions]
    normalized_refs = [normalize_text(ref) for ref in batch_refs]
    
    for pred, ref, norm_pred, norm_ref in zip(predictions, batch_refs, normalized_predictions, normalized_refs):
        print(f"Prediction: \t{norm_pred}")
        print(f"Reference: \t{norm_ref}")

    # Calculate metrics using normalized text
    batch_wer = wer(normalized_refs, normalized_predictions)
    batch_cer = cer(normalized_refs, normalized_predictions)
    print(f"Batch WER: {batch_wer:.4f}")
    print(f"Batch CER: {batch_cer:.4f}")
    all_preds.extend(normalized_predictions)
    all_refs.extend(normalized_refs)

    # Periodic cache clearing
    if num_batches % 10 == 0:
        torch.cuda.empty_cache()

total_wer = wer(all_refs, all_preds)
total_cer = cer(all_refs, all_preds)
print(f"\nAverage WER: {total_wer:.4f}")
print(f"\nAverage CER: {total_cer:.4f}")
results_df = pd.DataFrame({
    'reference': all_refs,
    'prediction': all_preds
})
results_df.to_csv('whisper_nst_results.csv', index=False, encoding='utf-8')
print("\nResults saved to whisper_nst_results.csv")