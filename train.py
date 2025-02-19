import functools
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup

from dataset.audio_dataset import AudioChatDataset, collate_fn
from models.audio_llm import AudioLLMModel
from utils import (
    load_checkpoint,
    print_parameter_count,
    run_evaluation,
    save_checkpoint,
)

# set up a module-level logger
logger = logging.getLogger(__name__)

def setup_wandb(runtime_config: DictConfig) -> None:
    wandb.init(
        entity="borkeren",
        project="multimodal-alignment",
        name=f"model-{runtime_config.model.lm_model_name.split('/', maxsplit=1)[-1]}-processor-{runtime_config.model.encoder_model_name.split('/', maxsplit=1)[-1]}-dataset-{runtime_config.dataset.name}-at-{datetime.now()}",
        config=dict(runtime_config)
    )

def create_dataset_and_loader(cfg, model, processor):

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dataset = AudioChatDataset(
        lm_model=cfg.model.lm_model_name,
        dataset=cfg.dataset.name,
        split=cfg.dataset.split,
        tokenizer=model.tokenizer,
        processor=processor,
        num_audios=cfg.dataset.num_audios if hasattr(cfg.dataset, 'num_audios') else 1
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.training.batch_size, 
        collate_fn=functools.partial(collate_fn, tokenizer=model.tokenizer), 
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True
    )
    return dataset, dataloader

def train(model: torch.nn.Module, cfg: DictConfig) -> None:
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if "w2v-bert" in cfg.model.encoder_model_name:
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained(cfg.model.encoder_model_name)
    else:
        processor = AutoProcessor.from_pretrained(cfg.model.encoder_model_name)
    
    # Initialize dataset with num_audios=1
    if not hasattr(cfg.dataset, 'num_audios'):
        cfg.dataset.num_audios = 1
    
    dataset, dataloader = create_dataset_and_loader(cfg, model, processor)
    
    test_dataset = AudioChatDataset(
        lm_model=cfg.model.lm_model_name,
        dataset=cfg.dataset.name,
        split="test.clean",
        tokenizer=model.tokenizer,
        processor=processor,
        num_audios=1
    )
    
    print_parameter_count(model, logger)
    
    trainable_params = [
        {"params": list(model.projector.parameters())},
    ]
    
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.training.lr)
    
    # Calculate total number of training steps
    total_steps = len(dataloader) * cfg.training.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=total_steps
    )

    global_step = 0
    if cfg.training.resume_from_checkpoint:
        global_step, _ = load_checkpoint(
            [model.projector], optimizer, scheduler,
            cfg.training.resume_checkpoint_path, logger
        )
        logger.info(f"Resuming training from step {global_step}")
    
    # Loss monitoring variables
    loss_window = []
    window_size = 100  # Monitor average loss over last 100 steps
    loss_threshold = 1.0  # Threshold to trigger num_audios increase
    
    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.monotonic()
        batch_start_time = time.monotonic()
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            
            if isinstance(batch["audio_features"][0], dict):
                audio_features = [{
                    'input_features': af["input_features"].to("cuda", torch.bfloat16, non_blocking=True),
                    'attention_mask': af["attention_mask"].to("cuda", non_blocking=True)
                } for af in batch["audio_features"]]
            else:
                audio_features = [af.to("cuda", torch.bfloat16, non_blocking=True) 
                                for af in batch["audio_features"]]
            
            labels = batch["labels"].to("cuda", non_blocking=True)
            alt_input_ids = batch["alt_input_ids"].to("cuda", non_blocking=True)
            alt_labels = batch["alt_labels"].to("cuda", non_blocking=True)
            attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
            insertion_indices = batch["insertion_indices"]
            
            outputs = model(
                input_ids=input_ids,
                audio_features=audio_features,
                labels=labels,
                alt_input_ids=alt_input_ids,
                alt_labels=alt_labels,
                attention_mask=attention_mask,
                insertion_indices=insertion_indices
            )
            
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # # update loss monitoring
            loss_value = loss.item()
            # loss_window.append(loss_value)
            # if len(loss_window) > window_size:
            #     loss_window.pop(0)
            
            # # check if we should increase num_audios
            # if (len(loss_window) == window_size and 
            #     sum(loss_window) / window_size < loss_threshold and 
            #     cfg.dataset.num_audios == 1):
                
            #     logger.info(f"Average loss {sum(loss_window) / window_size:.4f} below threshold {loss_threshold}. Increasing num_audios to 4.")
            #     cfg.dataset.num_audios = 4
                
            #     # create new dataset and dataloader with updated num_audios
            #     dataset, dataloader = create_dataset_and_loader(cfg, model, processor)
                
            #     # recalculate total steps and update scheduler
            #     remaining_epochs = cfg.training.num_epochs - epoch - 1
            #     remaining_steps = len(dataloader) * remaining_epochs
            #     scheduler = get_linear_schedule_with_warmup(
            #         optimizer,
            #         num_warmup_steps=0,  # no warmup needed for continuation
            #         num_training_steps=remaining_steps
            #     )
                
            #     # clear loss window for new monitoring
            #     loss_window = []
            #     break  # restart epoch with new dataloader
            
            #total_loss += loss_value
            global_step += 1
            
            batch_time = time.monotonic() - batch_start_time
            
            if global_step % cfg.training.log_every_n_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                memory_allocated = torch.cuda.memory_allocated("cuda") / 1024**2
                memory_reserved = torch.cuda.memory_reserved("cuda") / 1024**2
                logger.info(
                    f"Step {global_step} | Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss_value:.4f} | Num Audios: {cfg.dataset.num_audios} | "
                    f"LR: {current_lr:.2e} | Batch Time: {batch_time:.2f}s | "
                    f"GPU Memory: Allocated {memory_allocated:.2f} MB, Reserved {memory_reserved:.2f} MB"
                )
                if cfg.wandb.enabled:
                    wandb.log({
                        "train/loss": loss_value,
                        "train/num_audios": cfg.dataset.num_audios,
                        "lr": current_lr,
                        "train/batch_time": batch_time
                    })
            
            if global_step % cfg.training.save_every_n_steps == 0:
                save_checkpoint(
                    [model.projector], optimizer, scheduler,
                    global_step, loss_value,
                    cfg.training.checkpoint_dir, logger
                )
                
            if global_step % cfg.training.evaluate_every_n_steps == 0:
                logger.info("\nRunning evaluation at step %d", global_step)
                run_evaluation(model, test_dataset, model.tokenizer, logger, cfg.eval.n_samples, cfg)
            
            batch_start_time = time.monotonic()
        
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.monotonic() - epoch_start_time
        avg_batch_time = epoch_time / len(dataloader)
        
        logger.info(
            f"Epoch {epoch+1} Summary | "
            f"Average Loss: {avg_loss:.4f} | "
            f"Num Audios: {cfg.dataset.num_audios} | "
            f"Epoch Time: {timedelta(seconds=int(epoch_time))} | "
            f"Average Batch Time: {avg_batch_time:.2f}s"
        )


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.cuda.set_per_process_memory_fraction(0.95, device=torch.cuda.current_device())
    # log the Hydra configuration for reproducibility
    logger.info("Hydra Config:\n" + str(cfg))
    # initialize components using config values from Hydra

    model = AudioLLMModel(cfg)

    # freeze non-trainable components
    for param in model.lm.parameters():
        param.requires_grad = False

    if cfg.wandb.enabled:
        setup_wandb(cfg)
    
    train(model, cfg)

if __name__ == "__main__":
    main()