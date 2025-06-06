import torch
import torchaudio
import transformers
from jiwer import wer, cer
from tqdm import tqdm
import re
from torch.utils.data import DataLoader, Dataset
from dataset.load_coral import coral_dataset_dict
import numpy as np
from benchmark.bootstrap import bootstrap_metrics

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

class CoralDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=48000, 
            new_freq=16000
        ).to(torch.float32)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio_array = torch.tensor(sample["audio"]["array"], dtype=torch.float32)
        resampled_audio = self.resampler(audio_array).numpy()
        return resampled_audio, sample["text"].lower()

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

def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)
    global resampler
    resampler = torchaudio.transforms.Resample(
        orig_freq=48000, 
        new_freq=16000
    ).to(torch.float32)

test_dataset = coral_dataset_dict["test"]

print(test_dataset)
test_dataset = CoralDatasetWrapper(test_dataset)

loader = DataLoader(
    test_dataset,
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=4,
    collate_fn=collate_fn,
    worker_init_fn=worker_init_fn,
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

    # Periodic cache clearing
    if num_batches % 10 == 0:
        print(f"WER so far: {np.mean(wers):.4f} +- {np.std(wers):.4f}")
        print(f"CER so far: {np.mean(cers):.4f} +- {np.std(cers):.4f}")
        torch.cuda.empty_cache()


# Final metricsw
wer_dict = bootstrap_metrics(wers, num_iterations=1000)
cer_dict = bootstrap_metrics(cers, num_iterations=1000)

print("Final WER / CER:")
print(f"WER: {wer_dict['mean']:.4f} +- {wer_dict['std']:.4f} | CI: {wer_dict['confidence_interval']}")
print(f"CER: {cer_dict['mean']:.4f} +- {cer_dict['std']:.4f} | CI: {cer_dict['confidence_interval']}")