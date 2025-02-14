import os
import torch
import torchaudio
import transformers
from jiwer import wer, cer
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import concurrent.futures
from functools import partial
from dataset.load_coral import coral_dataset_dict
import torch.nn.functional as F

# Optimize CUDA operations
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(0.7, device=torch.cuda.current_device())
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

model_name = "facebook/seamless-m4t-v2-large"
processor = transformers.AutoProcessor.from_pretrained(model_name)

class CoralTestDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=48000, 
            new_freq=16000
        ).to(torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        audio_array = torch.tensor(sample["audio"]["array"], dtype=torch.float32)
        resampled_audio = self.resampler(audio_array).numpy()
        return resampled_audio, sample["text"].lower()

def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)
    global resampler
    resampler = torchaudio.transforms.Resample(
        orig_freq=48000, 
        new_freq=16000
    ).to(torch.float32)

def collate_fn(batch):
    audios, refs = zip(*batch)
    inputs = processor(
        audios=list(audios), 
        return_tensors="pt", 
        sampling_rate=16000, 
        padding=True
    )
    return inputs, list(refs)

test_data = coral_dataset_dict["test"]
test_dataset = CoralTestDataset(test_data)

loader = DataLoader(
    test_dataset,
    batch_size=512,
    num_workers=32,
    pin_memory=True,
    prefetch_factor=2,
    collate_fn=collate_fn,
    worker_init_fn=worker_init_fn,
)

def load_model(device):
    model = transformers.SeamlessM4Tv2ForSpeechToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        use_cache=True
    )
    model = model.to(device, non_blocking=True)
    model = torch.compile(
        model,
        mode="max-autotune",
        fullgraph=True
    )
    model.eval()
    return model

# Load models with optimized initialization
model0 = load_model("cuda:0")
model1 = load_model("cuda:1")

# Create CUDA streams for parallel execution
stream0 = torch.cuda.Stream(device="cuda:0")
stream1 = torch.cuda.Stream(device="cuda:1")

def generate_chunk(model, chunk, device, stream):
    with torch.cuda.stream(stream):
        chunk = chunk.to(device, non_blocking=True)
        with torch.inference_mode(), torch.amp.autocast(dtype=torch.bfloat16):
            generated = model.generate(
                chunk,
                tgt_lang="dan",
                num_beams=1,
                length_penalty=1.0,
                use_cache=True
            )
            return generated.cpu()

def process_batch_predictions(tokens0, tokens1):
    # Process predictions in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future0 = executor.submit(processor.batch_decode, tokens0, skip_special_tokens=True)
        future1 = executor.submit(processor.batch_decode, tokens1, skip_special_tokens=True)
        predictions0 = future0.result()
        predictions1 = future1.result()
    return predictions0 + predictions1

total_wer = 0.0
total_cer = 0.0
num_batches = 0

# Pre-compile generate functions with specific devices and streams
generate0 = partial(generate_chunk, model0, device=torch.device("cuda:0"), stream=stream0)
generate1 = partial(generate_chunk, model1, device=torch.device("cuda:1"), stream=stream1)

for inputs, batch_refs in tqdm(loader, desc="Processing batches"):
    # Pin memory and split batch
    input_features = inputs["input_features"].pin_memory()
    splits = torch.chunk(input_features, 2, dim=0)
    
    # Process chunks in parallel using pre-compiled functions
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future0 = executor.submit(generate0, splits[0])
        future1 = executor.submit(generate1, splits[1])
        tokens0 = future0.result()
        tokens1 = future1.result()

    predictions = process_batch_predictions(tokens0, tokens1)
    
    batch_wer = wer(batch_refs, predictions)
    batch_cer = cer(batch_refs, predictions)
    total_wer += batch_wer
    total_cer += batch_cer
    num_batches += 1

    if num_batches % 10 == 0:
        torch.cuda.empty_cache()

average_wer = total_wer / num_batches
average_cer = total_cer / num_batches
print(f"\nAverage WER: {average_wer:.4f}")
print(f"\nAverage CER: {average_cer:.4f}")