import torch
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from liger_kernel.transformers import AutoLigerKernelForCausalLM # for efficient long context


dataset = load_dataset("trl-lib/Capybara", split="train").select(range(2**8)) # small dataset for testing

model = AutoLigerKernelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda",
    use_cache=False, # no cache for training
)

training_args = SFTConfig(
    output_dir="HuggingFaceTB/SmolLM2-360M-SFT",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    max_seq_length=4096,
    max_grad_norm=1.0,
    logging_steps=1,
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset,
)

result = trainer.train()
