import os
from benchmark.bootstrap import bootstrap_metrics

# Check if the file exists and throw an error if it does
if os.path.exists("generations/whisper_librispeech.txt"):
    raise FileExistsError("The file 'generations/whisper_librispeech.txt' already exists.")

import torch
import transformers
from tqdm import tqdm
import json
from torch.utils.data import DataLoader, Dataset
from dataset.load_librispeech import librispeech_dataset_dict
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(0.9, device=torch.cuda.current_device())
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Use OpenAI's Whisper model
model_name = "openai/whisper-large-v3"
processor = transformers.WhisperProcessor.from_pretrained(model_name)

class LibrispeechDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio_array = sample["audio"]["array"] 
        return audio_array, sample["text"].lower(), sample["id"]

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
# test_dataset = test_dataset.filter(filter_difficult_examples)

print(test_dataset)
test_dataset = LibrispeechDatasetWrapper(test_dataset)

loader = DataLoader(
    test_dataset,
    batch_size=16,
    #num_workers=8,
    pin_memory=True,
    #prefetch_factor=4,
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

forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

generated_dict = {}

for num_batches, (inputs, _, record_ids) in enumerate(tqdm(loader, desc="Processing batches")):
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

    generated_dict.update(dict(zip(record_ids, predictions)))

with open("generations/whisper_librispeech.txt", "w+") as f:
    f.write(json.dumps(generated_dict, indent=4))