import functools
import os
import torchaudio
from models.SparkTTS import SparkTTS
from pathlib import Path
import hydra
from omegaconf import DictConfig
from transformers import AutoProcessor
import torch
from torch.utils.data import DataLoader
from peft import inject_adapter_in_model, LoraConfig
from dataset.audio_dataset import AudioChatDataset, collate_fn
from models.audio_llm import AudioLLMModel
from utils import (
    load_checkpoint,
    print_parameter_count,
    run_evaluation,
    save_checkpoint,
    calculate_eval_loss,
    run_evaluation_with_loss
)

@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Load models
    model = AudioLLMModel(cfg)

    model_wrap = SparkTTS(os.environ["HF_HOME"], True, "cuda")
    for param in model_wrap.model.parameters():
        param.requires_grad = False
    model_wrap.model.eval()

    processor = AutoProcessor.from_pretrained(cfg.model.encoder_model_name)
    dataset = AudioChatDataset(
        lm_model="google/gemma-2-2b-it",
        dataset="librispeech",
        processor=processor,
        tokenizer=model.tokenizer,
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        collate_fn=functools.partial(collate_fn, tokenizer=model.tokenizer, T=dataset.T), 
        shuffle=False,
    )

    for batch in dataloader:
        print(batch)
        break

if __name__ == "__main__":
    main()