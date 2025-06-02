import torch
import transformers
from jiwer import wer, cer
from tqdm import tqdm
import re
from torch.utils.data import DataLoader, Dataset
from dataset.load_librispeech import librispeech_dataset_dict
from dataset.audio_dataset import filter_difficult_examples
import numpy as np
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
    
    # Define number mappings for English
    basic_numbers = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
        "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19
    }

    tens = {
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
    }

    multipliers = {
        "hundred": 100, "thousand": 1000, "million": 1000000,
        "billion": 1000000000
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
                    rest_result = rest_text # normalize_numbers_danish(rest_text)
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
model_name = "syvai/hviske-v2"
processor = transformers.WhisperProcessor.from_pretrained(model_name)

class LibrispeechDatasetWrapper(Dataset):
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

test_dataset = librispeech_dataset_dict["test"]
print(test_dataset)
test_dataset = test_dataset.filter(filter_difficult_examples)

print(test_dataset)
test_dataset = LibrispeechDatasetWrapper(test_dataset)

loader = DataLoader(
    test_dataset,
    batch_size=32,
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

wers = []
cers = []

for num_batches, (inputs, batch_refs) in enumerate(tqdm(loader, desc="Processing batches")):
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

        # Calculate metrics using normalized text
        single_wer = wer(norm_ref, norm_pred)
        single_cer = cer(norm_ref, norm_pred)
        wers.append(single_wer)
        cers.append(single_cer)
    
    print(f"Prediction: \t{norm_pred}")
    print(f"Reference: \t{norm_ref}")

    # Periodic cache clearing
    if num_batches % 10 == 0:
        print("WER / CER so far:")
        print(f"WER: {np.mean(wers):.4f} +- {np.std(wers):.4f}")
        print(f"CER: {np.mean(cers):.4f} +- {np.std(cers):.4f}")
        torch.cuda.empty_cache()


print("Final WER / CER:")
print(f"WER: {np.mean(wers):.4f} +- {np.std(wers):.4f}")
print(f"CER: {np.mean(cers):.4f} +- {np.std(cers):.4f}")