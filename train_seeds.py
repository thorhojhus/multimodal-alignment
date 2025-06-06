import functools
import logging
import time
from datetime import timedelta
from pathlib import Path
from config import AppConfig
import numpy
import hydra
import torch
import wandb
import random
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from peft import inject_adapter_in_model, LoraConfig
from dataset.audio_dataset import AudioChatDataset, collate_fn
from models.audio_llm import AudioLLMModel
from utils import (
    print_parameter_count,
    run_evaluation,
    calculate_eval_loss,
    run_evaluation_with_loss
)

import warnings

# Also use the warnings filter as backup
warnings.filterwarnings("ignore", message="words count mismatch")

# set up a module-level logger
logger = logging.getLogger(__name__)
logging.getLogger('phonemizer').setLevel(logging.ERROR)

def setup_wandb(runtime_config: AppConfig, iter: int) -> None:

    wandb.init(
        entity="borkeren",
        project="multimodal-alignment"
                if not runtime_config.wandb.project
                else runtime_config.wandb.project,
        name=f"model-{runtime_config.model.lm_model_name.split('/', maxsplit=1)[-1]}-processor-{runtime_config.model.encoder_model_name.split('/', maxsplit=1)[-1]}-dataset-{runtime_config.dataset.name}-{iter}",
        config=dict(runtime_config)
    )

def train(model: torch.nn.Module, cfg: AppConfig, num_epochs: int) -> None:
    # create checkpoint directory
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # create dataset and processor
    if ("w2v-bert" in cfg.model.encoder_model_name):
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained(cfg.model.encoder_model_name, padding_value=0.0)
    elif ("BiCodec" in cfg.model.encoder_model_name):
        from transformers import Wav2Vec2FeatureExtractor
        processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    elif ("mimi" in cfg.model.encoder_model_name) or ("kyutai/mimi" == cfg.model.encoder_model_name):
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    else:
        try:
            processor = AutoProcessor.from_pretrained(cfg.model.encoder_model_name)
        except Exception:
            from transformers import AutoFeatureExtractor
            processor = AutoFeatureExtractor.from_pretrained(cfg.model.encoder_model_name)
    
    dataset = AudioChatDataset(
        lm_model=cfg.model.lm_model_name,
        dataset=cfg.dataset.name,
        split=cfg.dataset.split,
        tokenizer=model.tokenizer,
        processor=processor,
        T=cfg.model.token_length,
        config=cfg,
    )

    test_dataset = AudioChatDataset(
        lm_model=cfg.model.lm_model_name,
        dataset=cfg.dataset.name,
        split=cfg.dataset.test_split,
        tokenizer=model.tokenizer,
        processor=processor,
        T=cfg.model.token_length,
        config=cfg,
    )

    trainable_params = []

    if cfg.model.train_encoder_wav2vec:
        model.audio_encoder.audio_encoder.gradient_checkpointing_enable()
        for param in model.audio_encoder.audio_encoder.encoder.parameters():
            param.requires_grad = True

        trainable_params.append({"params": list(model.audio_encoder.audio_encoder.encoder.parameters())})
        logger.info("Training encoder parameters")

    if cfg.model.train_encoder_full:
        model.audio_encoder.audio_encoder.gradient_checkpointing_enable()
        for param in model.audio_encoder.audio_encoder.parameters():
            param.requires_grad = True

        trainable_params.append({"params": list(model.audio_encoder.audio_encoder.parameters())})
        logger.info("Training full encoder parameters")
            
    if hasattr(model, 'projector') and model.projector is not None:
        # Standard mode - training the projector
        trainable_params.append({"params": list(model.projector.parameters())})
        logger.info("Training projector parameters")

    if not trainable_params:
        raise ValueError("No trainable parameters found. Check model configuration.")

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.training.lr)
    
    print_parameter_count(model, logger)

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.training.batch_size, 
        collate_fn=functools.partial(collate_fn, tokenizer=model.tokenizer, T=dataset.T), 
        shuffle=True,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=True
    )
    
    # Calculate effective total number of training steps with gradient accumulation
    total_steps = len(dataloader) * cfg.training.num_epochs // cfg.training.gradient_accumulation_steps
    
    # create the scheduler using warmup_steps from config
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=total_steps
    )

    global_step = 0
    
    # Zero gradients at the beginning of each epoch
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.monotonic()
        batch_start_time = time.monotonic()

        for batch_idx, batch in enumerate(dataloader):
            # move data to GPU
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            if isinstance(batch["audio_features"], dict):
                if "input_features" in batch["audio_features"]:
                    # for wav2vec-bert/seamless models
                    audio_features = {
                        'input_features': batch["audio_features"]["input_features"].to("cuda", torch.bfloat16, non_blocking=True),
                        'attention_mask': batch["audio_features"]["attention_mask"].to("cuda", non_blocking=True)
                    }
                elif "input_values" in batch["audio_features"]:
                    # for Mimi models
                    audio_features = {
                        'input_values': batch["audio_features"]["input_values"].to("cuda", torch.bfloat16, non_blocking=True)
                    }
                else:
                    # fallback
                    audio_features = {k: v.to("cuda", torch.bfloat16, non_blocking=True) for k, v in batch["audio_features"].items()}
            else:
                # for other models (Whisper, CLAP, wav2vec)
                audio_features = batch["audio_features"].to("cuda", torch.bfloat16, non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)
            alt_input_ids = batch["alt_input_ids"].to("cuda", non_blocking=True)
            alt_labels = batch["alt_labels"].to("cuda", non_blocking=True)
            alt_attention_mask = batch["alt_attention_mask"].to("cuda", non_blocking=True)
            attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
            insertion_indices = batch["insertion_indices"]
            text_ids = batch["text_ids"].to("cuda", non_blocking=True)
            # forward pass
            outputs = model(
                input_ids=input_ids,
                audio_features=audio_features,
                labels=labels,
                alt_input_ids=alt_input_ids,
                alt_labels=alt_labels,
                alt_attention_mask=alt_attention_mask,
                attention_mask=attention_mask,
                insertion_indices=insertion_indices,
                text_ids=text_ids
            )
            # backward pass
            loss = outputs.loss
            # scale the loss by the number of gradient accumulation steps
            loss = loss / cfg.training.gradient_accumulation_steps
            loss.backward()
            
            # only update weights after accumulating gradients for the specified number of steps
            if (batch_idx + 1) % cfg.training.gradient_accumulation_steps == 0:
                # clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    [p for group in trainable_params for p in group["params"]], 
                    max_norm=cfg.training.gradient_clip_norm
                )
                
                # optimizer and scheduler steps
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Each gradient accumulation cycle counts as one step for logging and evaluation
                global_step += 1
            
            total_loss += loss.item() * cfg.training.gradient_accumulation_steps
            
            # calculate batch time
            batch_time = time.monotonic() - batch_start_time
            
            # for logging, determine if this is a gradient update step
            is_update_step = (batch_idx + 1) % cfg.training.gradient_accumulation_steps == 0
            
            # Only log on designated log steps and after parameter updates
            if global_step > 0 and global_step % cfg.training.log_every_n_steps == 0 and is_update_step:
                current_lr = scheduler.get_last_lr()[0]
                memory_allocated = torch.cuda.memory_allocated("cuda") / 1024**2
                memory_reserved = torch.cuda.memory_reserved("cuda") / 1024**2
                #torch.cuda.empty_cache()
                logger.info(
                    f"Step {global_step} | Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss.item() * cfg.training.gradient_accumulation_steps:.4f} | "
                    f"LR: {current_lr:.2e} | Batch Time: {batch_time:.2f}s | "
                    f"GPU Memory: Allocated {memory_allocated:.2f} MB, Reserved {memory_reserved:.2f} MB"
                )
                if cfg.wandb.enabled:
                    wandb.log({
                        "train/loss": loss.item() * cfg.training.gradient_accumulation_steps, 
                        "lr": current_lr, 
                        "train/batch_time": batch_time
                    })
            

            # run evaluation and loss calculation if needed
            # Only run evaluation after parameter updates and on evaluation steps
            if global_step > 0 and global_step % cfg.training.evaluate_every_n_steps == 0 and is_update_step:
                logger.info("\nRunning evaluation at step %d", global_step)
                
                # calculate evaluation loss
                eval_loss = calculate_eval_loss(
                    model=model,
                    test_dataset=test_dataset,
                    tokenizer=model.tokenizer,
                    logger=logger,
                    n_samples=cfg.eval.n_samples,
                    batch_size=cfg.eval.batch_size or cfg.training.batch_size
                )
                
                # run WER/CER evaluation
                wer, cer, corrected_wer_list = run_evaluation(
                    model=model,
                    test_dataset=test_dataset,
                    tokenizer=model.tokenizer,
                    logger=logger,
                    n_samples=cfg.eval.n_samples,
                    config=cfg,
                    batch_size=cfg.eval.batch_size or cfg.training.batch_size
                )
                
                logger.info(
                    f"Step {global_step} Evaluation | "
                    f"Eval Loss: {eval_loss:.4f} | "
                    f"WER: {wer:.2f} (C: {(sum(corrected_wer_list) / len(corrected_wer_list)):0.4f}) | CER: {cer:.2f}"
                )
                if cfg.wandb.enabled:
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/wer": wer,
                        "eval/wer_corrected": sum(corrected_wer_list) / len(corrected_wer_list),
                        "eval/cer": cer
                    })

            # reset batch timer
            batch_start_time = time.monotonic()
        
        # calculate epoch statistics
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.monotonic() - epoch_start_time
        avg_batch_time = epoch_time / len(dataloader)
        
        # end of epoch evaluation
        logger.info("\nRunning end of epoch evaluation")
        eval_loss, wer, cer, corrected_wer_list = run_evaluation_with_loss(
            model=model, 
            test_dataset=test_dataset, 
            tokenizer=model.tokenizer, 
            logger=logger, 
            n_samples=None, 
            config=cfg
        )   
        
        logger.info(
            "Epoch Summary | "
            f"Average Train Loss: {avg_loss:.4f} | "
            f"Eval Loss: {eval_loss:.4f} | "
            f"WER: {wer:.2f} (C: {(sum(corrected_wer_list) / len(corrected_wer_list)):0.4f}) | CER: {cer:.2f} | "
            f"Epoch Time: {timedelta(seconds=int(epoch_time))} | "
            f"Average Batch Time: {avg_batch_time:.2f}s"
        )
        if cfg.wandb.enabled:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "eval/epoch_loss": eval_loss,
                "eval/epoch_wer": wer,
                "eval/epoch_wer_corrected": sum(corrected_wer_list) / len(corrected_wer_list),
                "eval/epoch_cer": cer
            })
        
    wandb.finish()
    return avg_loss, wer, cer


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(hydra_conf: DictConfig):
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.cuda.set_per_process_memory_fraction(0.95, device=torch.cuda.current_device())
    # log the Hydra configuration for reproducibility
    logger.info("Hydra Config:\n" + str(hydra_conf))
    # initialize components using config values from Hydra

    cfg = AppConfig(**hydra_conf)
    losses, wers, cers = [], [], []
    for i in range(1):
        torch.manual_seed(cfg.training.seed or 451)
        numpy.random.seed(cfg.training.seed or 451)
        torch.cuda.manual_seed(cfg.training.seed or 451)
        random.seed(cfg.training.seed or 451)
        torch.backends.cudnn.deterministic = True

        model = AudioLLMModel(cfg)

        lora_config = LoraConfig(
            r=256,
            lora_alpha=32,
            target_modules="all-linear",
            lora_dropout=0.1,
            bias="none",
            use_dora=False,
            use_rslora=True,
        )

        if cfg.model.train_llm:
            model.lm = inject_adapter_in_model(lora_config, model.lm)

        if cfg.model.train_encoder:
            model.audio_encoder.audio_encoder = inject_adapter_in_model(lora_config, model.audio_encoder.audio_encoder)
        
        if cfg.wandb.enabled:
            setup_wandb(cfg, cfg.training.run_count)

        logger.info(f"Starting training on seed [{451 + i}]")
        final_loss, final_wer, final_cer = train(model, cfg, 1)

        losses.append(final_loss)
        wers.append(final_wer)
        cers.append(final_cer)
        logger.info(f"[{i}] Loss: {final_loss} | WER: {final_wer} | CER: {final_cer}")

        torch.cuda.empty_cache()

    logger.info(f"Losses: {losses} | WERs: {wers} | CERs: {cers}")

if __name__ == "__main__":
    main()
