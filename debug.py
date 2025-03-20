import functools
import logging

import hydra
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from dataset.audio_dataset import AudioChatDataset, collate_fn
from models.audio_llm import AudioLLMModel
from utils import (
    run_evaluation
)

@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg):
    processor = AutoProcessor.from_pretrained(cfg.model.encoder_model_name)
    model = AudioLLMModel(cfg)

    dataset = AudioChatDataset(
        lm_model="google/gemma-2-2b-it",
        dataset="coral",
        split="train",
        tokenizer=model.tokenizer,
        processor=processor,
        T=30,
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=3, 
        collate_fn=functools.partial(collate_fn, tokenizer=model.tokenizer, T=dataset.T, pad=cfg.training.do_ctx_utilization), 
        shuffle=True,
        num_workers=0
    )

    for batch_idx, batch in enumerate(dataloader):
        # Prepare data
        audio_features = batch["audio_features"].to("cuda", torch.bfloat16, non_blocking=True)
        insertion_indices = batch["insertion_indices"]
        
        if cfg.training.do_ctx_utilization:
            input_ids = [x.to("cuda", non_blocking=True) for x in batch["input_ids"]]
            labels = [x.to("cuda", non_blocking=True) for x in batch["labels"]]
            attention_mask = [x.to("cuda", non_blocking=True) for x in batch["attention_mask"]]
            insertion_indices[0] = insertion_indices[0].item()
            for i in range(1, len(insertion_indices)):
                insertion_indices[i] = insertion_indices[i].item() + torch.numel(torch.cat((*input_ids[:i],)))
            
            input_ids = torch.cat((*input_ids,)).to("cuda", non_blocking=True)
            attention_mask = torch.cat((*attention_mask,)).to("cuda", non_blocking=True)
            labels = torch.cat((*labels,)).to("cuda", non_blocking=True)
        else:
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)
            attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)

        alt_input_ids, alt_attention_mask, alt_labels = None, None, None

        # KL loss specifics
        if cfg.training.loss == "kl_loss":
            if cfg.training.do_ctx_utilization:
                alt_input_ids = torch.cat((*batch["alt_input_ids"],)).to("cuda", non_blocking=True)
                alt_attention_mask = torch.cat((*batch["alt_attention_mask"],)).to("cuda", non_blocking=True)
                alt_labels = torch.cat((*batch["alt_labels"],)).to("cuda", non_blocking=True)
            else:
                alt_input_ids = batch["alt_input_ids"].to("cuda", non_blocking=True)
                alt_attention_mask = batch["alt_attention_mask"].to("cuda", non_blocking=True)
                alt_labels = batch["alt_labels"].to("cuda", non_blocking=True)

        # forward pass
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            labels=labels,
            insertion_indices=insertion_indices,
            alt_input_ids=alt_input_ids,
            alt_attention_mask=alt_attention_mask,
            alt_labels=alt_labels,
            combined=cfg.training.do_ctx_utilization,
        )

        run_evaluation(model, dataset, model.tokenizer, logging.getLogger(__name__), 2, cfg, batch_size=2)
        

if __name__ == "__main__":
    main()