import torch
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

# Workaround for CUDA using RAM as VRAM in WSL
#torch.cuda.set_per_process_memory_fraction(1.0, 0)

# Load the dataset and subset to 2000 samples
dataset = load_dataset("trl-lib/Capybara", split="train")

# Define training arguments
training_args = SFTConfig(
    output_dir="Qwen/Qwen2.5-0.5B-SFT",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    max_seq_length=4096,
    use_liger=True,
    gradient_checkpointing=True,
    packing=True,
    tf32=True,
    model_init_kwargs={
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2" if torch.cuda.is_available() else "eager",
    }
)

# Initialize the trainer
trainer = SFTTrainer(
    args=training_args,
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
)

# Train the model
result = trainer.train()
print_summary(result)

