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

# Set various CUDA and cudnn settings
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(0.9, device=torch.cuda.current_device())
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def normalize_numbers_danish(text):
    """
    Convert Danish number words to digits, handling complex number phrases.
    """
    # Replace common composite forms first
    text = text.replace("og ", " ")  # Remove connecting "og" between numbers
    
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
    
    # Process text through multiple passes
    words = text.split()
    result = []
    i = 0
    
    while i < len(words):
        current_word = words[i]
        
        # Check for compound numbers (e.g., "treogtyve")
        if "og" in current_word and len(current_word) > 2:
            parts = current_word.split("og")
            if len(parts) == 2 and parts[0] in basic_numbers and parts[1] in tens:
                result.append(str(basic_numbers[parts[0]] + tens[parts[1]]))
                i += 1
                continue
        
        # Process number phrases
        if current_word in basic_numbers:
            value = basic_numbers[current_word]
            
            # Look ahead for multipliers
            if i + 1 < len(words) and words[i+1] in multipliers:
                value *= multipliers[words[i+1]]
                i += 2
                
                # Check if we need to add more numbers
                if i < len(words):
                    next_val = 0
                    
                    # Try to process the rest as a number
                    rest_text = " ".join(words[i:])
                    rest_result = normalize_numbers_danish(rest_text)
                    if any(c.isdigit() for c in rest_result):
                        # Extract number from result
                        num_str = ''.join(c for c in rest_result if c.isdigit())
                        if num_str:
                            next_val = int(num_str)
                    
                    result.append(str(value + next_val))
                    break
                else:
                    result.append(str(value))
            else:
                result.append(str(value))
                i += 1
                
        elif current_word in tens:
            result.append(str(tens[current_word]))
            i += 1
            
        elif current_word in multipliers:
            # Handle standalone multipliers (rare but possible)
            result.append(str(multipliers[current_word]))
            i += 1
            
        else:
            # Not a number word, keep as is
            result.append(current_word)
            i += 1
    
    # Join and return
    return " ".join(result)

def normalize_text(text):
    """
    Normalize text to lowercase a-z, æ, ø, å, 0-9, and spaces only.
    Convert number words to digits and remove all other characters.
    """
    # Convert to lowercase
    text = text.lower()
    text = text.strip()
    
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
    batch_size=64,
    num_workers=16,
    pin_memory=True,
    prefetch_factor=8,
    collate_fn=collate_fn,
)

device = torch.device("cuda")

model = transformers.WhisperForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    use_cache=True
)
model = model.to(device)
model = torch.compile(
    model,
    mode="max-autotune",
    fullgraph=True
)
model.eval()

forced_decoder_ids = processor.get_decoder_prompt_ids(language="danish", task="transcribe")

total_wer = 0.0
total_cer = 0.0
num_batches = 0

all_preds = []
all_refs = []

for inputs, batch_refs in tqdm(loader, desc="Processing batches"):
    input_features = inputs["input_features"].to(device, torch.bfloat16, non_blocking=True)
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