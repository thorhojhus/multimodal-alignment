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

def normalize_text(text):
    """
    Normalize text to lowercase a-z, æ, ø, å, 0-9, and spaces only.
    Remove all other characters.
    """
    # Convert to lowercase
    text = text.lower()
    text = text.strip()
    
    # Keep only a-z, æ, ø, å, 0-9, and spaces
    text = re.sub(r'[^a-zæøå0-9 ]', '', text)
    
    return text

# Use OpenAI's Whisper model
model_name = "syvai/hviske-v2"
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