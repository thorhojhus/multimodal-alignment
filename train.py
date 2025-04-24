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
from peft import inject_adapter_in_model, LoraConfig
from dataset.audio_dataset import AudioChatDataset, collate_fn, add_gaussian_noise, apply_band_stop_augmentation, apply_pitch_shift_augmentation
from models.audio_llm import AudioLLMModel
from utils import (
    load_checkpoint,
    print_parameter_count,
    run_evaluation,
    save_checkpoint,
    calculate_eval_loss,
    run_evaluation_with_loss
)

import warnings
import logging

# Also use the warnings filter as backup
warnings.filterwarnings("ignore", message="words count mismatch")

# set up a module-level logger
logger = logging.getLogger(__name__)
logging.getLogger('phonemizer').setLevel(logging.ERROR)

def setup_wandb(runtime_config: DictConfig) -> None:

    wandb.init(
        entity="borkeren",
        project="multimodal-alignment"
                if not runtime_config.wandb.get('project', None)
                else runtime_config.wandb.project,
        name=f"model-{runtime_config.model.lm_model_name.split('/', maxsplit=1)[-1]}-processor-{runtime_config.model.encoder_model_name.split('/', maxsplit=1)[-1]}-dataset-{runtime_config.dataset.name}-at-{datetime.now()}"
             if runtime_config.wandb.run_name == "default"
             else runtime_config.wandb.run_name,
        config=dict(runtime_config)
    )

def train(model: torch.nn.Module, cfg: DictConfig) -> None:
    # create checkpoint directory
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # initialize best evaluation loss for checkpoint saving
    best_eval_loss = float('inf')

    # create dataset and processor
    if ("w2v-bert" in cfg.model.encoder_model_name):
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained(cfg.model.encoder_model_name)
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

    match cfg.dataset.augment:
        case "gaussian":
            logger.info("Adding amplitude noise augmentation")
            dataset.dataset.set_transform(add_gaussian_noise)
        case "bandstop":
            logger.info("Applying band stop augmentation")
            dataset.dataset.set_transform(apply_band_stop_augmentation)
        case "pitchshift":
            logger.info("Adding pitch shift augmentation")
            dataset.dataset.set_transform(apply_pitch_shift_augmentation)
        case _:
            logger.info("Applying no data augmentations")
            pass

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

    if cfg.model.expand_embeds:
        # When using expand_embeds, we train only the embeddings for audio tokens
        logger.info("Using expand_embeds mode - training only the new audio token embeddings")
        
        trainable_params.append({"params": list(model.lm.get_input_embeddings().parameters())})
    
    if cfg.model.train_llm:
        print(model.lm)
        lora_params = []
        for name, param in model.lm.named_parameters():
            if ('lora' in name) and param.requires_grad:
                lora_params.append(param)
        
        # Append as a dictionary with "params" key to match other branches
        trainable_params.append({"params": lora_params})
        logger.info(f"Training LoRA parameters: {len(lora_params)} parameters")

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
    total_steps = len(dataloader) * cfg.training.num_epochs // cfg.training.get('gradient_accumulation_steps', 1)
    
    # create the scheduler using warmup_steps from config
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=total_steps
    )

    # load checkpoint if specified
    global_step = 0
    if cfg.training.resume_from_checkpoint:
        # determine what components to load
        load_components = [model.projector]
        if cfg.model.train_encoder_full:
            load_components.append(model.audio_encoder.audio_encoder)
            
        global_step, _ = load_checkpoint(
            load_components, optimizer, scheduler,
            cfg.training.resume_checkpoint_path, logger,
            load_lr_from_checkpoint=cfg.training.load_lr_from_checkpoint
        )
        logger.info(f"Resuming training from step {global_step}")
    
    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.monotonic()
        batch_start_time = time.monotonic()
        
        # Zero gradients at the beginning of each epoch
        optimizer.zero_grad()
        
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
            loss = loss / cfg.training.get('gradient_accumulation_steps', 1)
            loss.backward()
            
            # only update weights after accumulating gradients for the specified number of steps
            if (batch_idx + 1) % cfg.training.get('gradient_accumulation_steps', 1) == 0:
                # clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    [p for group in trainable_params for p in group["params"]], 
                    max_norm=cfg.training.get('gradient_clip_norm', 1.0)
                )
                
                # optimizer and scheduler steps
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Each gradient accumulation cycle counts as one step for logging and evaluation
                global_step += 1
            
            total_loss += loss.item() * cfg.training.get('gradient_accumulation_steps', 1)
            
            # calculate batch time
            batch_time = time.monotonic() - batch_start_time
            
            # for logging, determine if this is a gradient update step
            is_update_step = (batch_idx + 1) % cfg.training.get('gradient_accumulation_steps', 1) == 0
            
            # Only log on designated log steps and after parameter updates
            if global_step > 0 and global_step % cfg.training.log_every_n_steps == 0 and is_update_step:
                current_lr = scheduler.get_last_lr()[0]
                memory_allocated = torch.cuda.memory_allocated("cuda") / 1024**2
                memory_reserved = torch.cuda.memory_reserved("cuda") / 1024**2
                #torch.cuda.empty_cache()
                logger.info(
                    f"Step {global_step} | Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss.item() * cfg.training.get('gradient_accumulation_steps', 1):.4f} | "
                    f"LR: {current_lr:.2e} | Batch Time: {batch_time:.2f}s | "
                    f"GPU Memory: Allocated {memory_allocated:.2f} MB, Reserved {memory_reserved:.2f} MB"
                )
                if cfg.wandb.enabled:
                    wandb.log({
                        "train/loss": loss.item() * cfg.training.get('gradient_accumulation_steps', 1), 
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
        
        # save checkpoint at end of epoch (always latest, and best if applicable)
        is_best = False
        if epoch == 0 or eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            is_best = True
            
        # check if we need to save encoder weights too
        save_components = [model.projector]
        if cfg.model.train_encoder_full:
            save_components.append(model.audio_encoder.audio_encoder)
            
        save_checkpoint(
            save_components, optimizer, scheduler,
            global_step, eval_loss, 
            cfg.training.checkpoint_dir, logger, is_best
        )
            
        logger.info(
            f"Epoch {epoch+1} Summary | "
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
        setup_wandb(cfg)

    train(model, cfg)

if __name__ == "__main__":
    main()
