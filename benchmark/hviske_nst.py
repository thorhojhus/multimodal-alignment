import torch
import transformers
from jiwer import wer, cer
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from dataset.nst_da import nst_dataset_dict
import re

# Set various CUDA and cudnn settings
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(0.9, device=torch.cuda.current_device())
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Use OpenAI's Whisper model
model_name = "syvai/hviske-v2"
processor = transformers.WhisperProcessor.from_pretrained(model_name)

class NSTTestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample["audio"]["array"], sample["text"].lower()

def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)

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

test_data = nst_dataset_dict["test"]
test_dataset = NSTTestDataset(test_data)

loader = DataLoader(
    test_dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
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
model.eval()

forced_decoder_ids = processor.get_decoder_prompt_ids(language="danish", task="transcribe")

total_wer = 0.0
total_cer = 0.0
num_batches = 0

for inputs, batch_refs in tqdm(loader, desc="Processing batches"):

    input_features = inputs["input_features"].to(device, non_blocking=True)
    attention_mask = inputs["attention_mask"].to(device, non_blocking=True)


    # Use autocast in float16 mode
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = model.generate(
            input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_decoder_ids,
            num_beams=1,
            length_penalty=1.0,
            use_cache=True,
        )
    
    predictions: str = processor.batch_decode(generated)
    predictions = [re.sub(r"(<\|endoftext\|>)+$", "", pred) for pred in predictions]
    batch_wer = wer(batch_refs, predictions)
    batch_cer = cer(batch_refs, predictions)
    total_wer += batch_wer
    total_cer += batch_cer
    num_batches += 1

    # Periodic cache clearing
    if num_batches % 100 == 0:

        print(f"--- Processed {num_batches} batches ---")
        print(f"\tAverage WER so far: {total_wer / num_batches:.4f}")
        print(f"\tAverage CER so far: {total_cer / num_batches:.4f}")
        print(f"Last generated: {predictions[-1]}")
        print(f"Last reference: {batch_refs[-1]}")

        torch.cuda.empty_cache()

average_wer = total_wer / num_batches
average_cer = total_cer / num_batches
print(f"Average WER: {average_wer:.4f}")
print(f"Average CER: {average_cer:.4f}")
