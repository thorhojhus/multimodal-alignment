import functools
import logging
import time
import os
from datetime import datetime, timedelta
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoProcessor, get_linear_schedule_with_warmup
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

import warnings
import logging

# Also use the warnings filter as backup
warnings.filterwarnings("ignore", message="words count mismatch")

# set up a module-level logger
logger = logging.getLogger(__name__)
logging.getLogger('phonemizer').setLevel(logging.ERROR)

def setup_wandb(runtime_config: DictConfig, rank: int) -> None:
    if rank == 0:  # Only initialize wandb on the main process
        wandb.init(
            entity="borkeren",
            project="multimodal-alignment",
            name=f"model-{runtime_config.model.lm_model_name.split('/', maxsplit=1)[-1]}-processor-{runtime_config.model.encoder_model_name.split('/', maxsplit=1)[-1]}-dataset-{runtime_config.dataset.name}-at-{datetime.now()}"
                if runtime_config.wandb.run_name == "default"
                else runtime_config.wandb.run_name,
            config=dict(runtime_config)
        )

def train(model: torch.nn.Module, cfg: DictConfig, local_rank: int, distributed: bool) -> None:
    # create checkpoint directory (only on rank 0)
    if local_rank == 0:
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
        except:
            from transformers import AutoFeatureExtractor
            processor = AutoFeatureExtractor.from_pretrained(cfg.model.encoder_model_name)
    
    # Get the actual model if it's wrapped in DDP
    unwrapped_model = model.module if distributed else model
    
    dataset = AudioChatDataset(
        lm_model=cfg.model.lm_model_name,
        dataset=cfg.dataset.name,
        split=cfg.dataset.split,
        tokenizer=unwrapped_model.tokenizer,
        processor=processor,
        T=cfg.model.token_length,
        config=cfg,
    )
    
    test_dataset = AudioChatDataset(
        lm_model=cfg.model.lm_model_name,
        dataset=cfg.dataset.name,
        split=cfg.dataset.test_split,
        tokenizer=unwrapped_model.tokenizer,
        processor=processor,
        T=cfg.model.token_length,
        config=cfg,
    )
    
    # Set up distributed sampler for training
    if distributed:
        train_sampler = DistributedSampler(dataset)
    else:
        train_sampler = None

    trainable_params = []

    if cfg.model.expand_embeds:
        # When using expand_embeds, we train only the embeddings for audio tokens
        if local_rank == 0:
            logger.info("Using expand_embeds mode - training only the new audio token embeddings")
        
        trainable_params.append({"params": list(unwrapped_model.lm.get_input_embeddings().parameters())})
    
    if cfg.model.train_llm:
        if local_rank == 0:
            print(unwrapped_model.lm)
        lora_params = []
        for name, param in unwrapped_model.lm.named_parameters():
            if ('lora' in name) and param.requires_grad:
                lora_params.append(param)
        
        # Append as a dictionary with "params" key to match other branches
        trainable_params.append({"params": lora_params})
        if local_rank == 0:
            logger.info(f"Training LoRA parameters: {len(lora_params)} parameters")

    if cfg.model.train_encoder_wav2vec:
        unwrapped_model.audio_encoder.audio_encoder.gradient_checkpointing_enable()
        for param in unwrapped_model.audio_encoder.audio_encoder.encoder.parameters():
            param.requires_grad = True

        trainable_params.append({"params": list(unwrapped_model.audio_encoder.audio_encoder.encoder.parameters())})
        if local_rank == 0:
            logger.info("Training encoder parameters")

    if cfg.model.train_encoder_full:
        unwrapped_model.audio_encoder.audio_encoder.gradient_checkpointing_enable()
        for param in unwrapped_model.audio_encoder.audio_encoder.parameters():
            param.requires_grad = True

        trainable_params.append({"params": list(unwrapped_model.audio_encoder.audio_encoder.parameters())})
        if local_rank == 0:
            logger.info("Training full encoder parameters")
            
    if hasattr(unwrapped_model, 'projector') and unwrapped_model.projector is not None:
        # Standard mode - training the projector
        trainable_params.append({"params": list(unwrapped_model.projector.parameters())})
        if local_rank == 0:
            logger.info("Training projector parameters")

    if not trainable_params:
        raise ValueError("No trainable parameters found. Check model configuration.")

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.training.lr)
    
    if local_rank == 0:
        print_parameter_count(unwrapped_model, logger)

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.training.batch_size, 
        collate_fn=functools.partial(collate_fn, tokenizer=unwrapped_model.tokenizer, T=dataset.T), 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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
        if local_rank == 0:
            logger.info(f"Loading checkpoint from {cfg.training.resume_checkpoint_path}")
        # Load on all processes to ensure model is synchronized
        global_step, _ = load_checkpoint(
            [unwrapped_model.projector], optimizer, scheduler,
            cfg.training.resume_checkpoint_path, logger if local_rank == 0 else None,
            load_lr_from_checkpoint=cfg.training.load_lr_from_checkpoint
        )
        if local_rank == 0:
            logger.info(f"Resuming training from step {global_step}")
    
    for epoch in range(cfg.training.num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)  # Important for proper shuffling in distributed training
            
        model.train()
        total_loss = 0
        epoch_start_time = time.monotonic()
        batch_start_time = time.monotonic()
        
        # Zero gradients at the beginning of each epoch
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # move data to GPU
            input_ids = batch["input_ids"].to(f"cuda:{local_rank}", non_blocking=True)
            if isinstance(batch["audio_features"], dict):
                if "input_features" in batch["audio_features"]:
                    # for wav2vec-bert/seamless models
                    audio_features = {
                        'input_features': batch["audio_features"]["input_features"].to(f"cuda:{local_rank}", torch.bfloat16, non_blocking=True),
                        'attention_mask': batch["audio_features"]["attention_mask"].to(f"cuda:{local_rank}", non_blocking=True)
                    }
                elif "input_values" in batch["audio_features"]:
                    # for Mimi models
                    audio_features = {
                        'input_values': batch["audio_features"]["input_values"].to(f"cuda:{local_rank}", torch.bfloat16, non_blocking=True)
                    }
                else:
                    # fallback
                    audio_features = {k: v.to(f"cuda:{local_rank}", torch.bfloat16, non_blocking=True) for k, v in batch["audio_features"].items()}
            else:
                # for other models (Whisper, CLAP, wav2vec)
                audio_features = batch["audio_features"].to(f"cuda:{local_rank}", torch.bfloat16, non_blocking=True)
            labels = batch["labels"].to(f"cuda:{local_rank}", non_blocking=True)
            alt_input_ids = batch["alt_input_ids"].to(f"cuda:{local_rank}", non_blocking=True)
            alt_labels = batch["alt_labels"].to(f"cuda:{local_rank}", non_blocking=True)
            alt_attention_mask = batch["alt_attention_mask"].to(f"cuda:{local_rank}", non_blocking=True)
            attention_mask = batch["attention_mask"].to(f"cuda:{local_rank}", non_blocking=True)
            insertion_indices = batch["insertion_indices"]
            text_ids = batch["text_ids"].to(f"cuda:{local_rank}", non_blocking=True)
            
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
            
            # Only log on designated log steps and after parameter updates and only on the main process
            if local_rank == 0 and global_step > 0 and global_step % cfg.training.log_every_n_steps == 0 and is_update_step:
                current_lr = scheduler.get_last_lr()[0]
                memory_allocated = torch.cuda.memory_allocated(f"cuda:{local_rank}") / 1024**2
                memory_reserved = torch.cuda.memory_reserved(f"cuda:{local_rank}") / 1024**2
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
            # Only perform evaluation on the main process
            if local_rank == 0 and global_step > 0 and global_step % cfg.training.evaluate_every_n_steps == 0 and is_update_step:
                logger.info("\nRunning evaluation at step %d", global_step)
                
                # Save model to a temporary file that all processes can load from
                if distributed:
                    # Create temp model for evaluation using unwrapped model components
                    eval_model = AudioLLMModel(cfg)
                    eval_model.projector = unwrapped_model.projector
                    eval_model.lm = unwrapped_model.lm
                    eval_model.audio_encoder = unwrapped_model.audio_encoder
                    eval_model.to(f"cuda:{local_rank}")
                else:
                    eval_model = model
                
                # calculate evaluation loss
                eval_loss = calculate_eval_loss(
                    model=eval_model,
                    test_dataset=test_dataset,
                    tokenizer=unwrapped_model.tokenizer,
                    logger=logger,
                    n_samples=cfg.eval.n_samples,
                    batch_size=cfg.eval.batch_size or cfg.training.batch_size
                )
                
                # run WER/CER evaluation
                wer, cer = run_evaluation(
                    model=eval_model,
                    test_dataset=test_dataset,
                    tokenizer=unwrapped_model.tokenizer,
                    logger=logger,
                    n_samples=cfg.eval.n_samples,
                    config=cfg,
                    batch_size=cfg.eval.batch_size or cfg.training.batch_size
                )
                
                logger.info(
                    f"Step {global_step} Evaluation | "
                    f"Eval Loss: {eval_loss:.4f} | "
                    f"WER: {wer:.2f} | CER: {cer:.2f}"
                )
                if cfg.wandb.enabled:
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/wer": wer,
                        "eval/cer": cer
                    })
                
                # Put model back into training mode
                model.train()
                
                # If using distributed, clean up the temporary evaluation model
                if distributed:
                    del eval_model
                    torch.cuda.empty_cache()

            # reset batch timer
            batch_start_time = time.monotonic()
        
        # Synchronize all processes before end of epoch evaluation
        if distributed:
            dist.barrier()
        
        # Calculate epoch statistics
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.monotonic() - epoch_start_time
        avg_batch_time = epoch_time / len(dataloader)
        
        # End of epoch evaluation (only on main process)
        if local_rank == 0:
            logger.info("\nRunning end of epoch evaluation")
            
            # Create evaluation model from unwrapped model components if needed
            if distributed:
                eval_model = AudioLLMModel(cfg)
                eval_model.projector = unwrapped_model.projector
                eval_model.lm = unwrapped_model.lm
                eval_model.audio_encoder = unwrapped_model.audio_encoder
                eval_model.to(f"cuda:{local_rank}")
            else:
                eval_model = model
            
            eval_loss, wer, cer = run_evaluation_with_loss(
                model=eval_model,
                test_dataset=test_dataset, 
                tokenizer=unwrapped_model.tokenizer, 
                logger=logger, 
                n_samples=None, 
                config=cfg
            )
            
            # Save checkpoint at end of epoch (always latest, and best if applicable)
            is_best = False
            if epoch == 0 or eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                is_best = True
                
            save_checkpoint(
                [unwrapped_model.projector], optimizer, scheduler,
                global_step, eval_loss, 
                cfg.training.checkpoint_dir, logger, is_best
            )
                
            logger.info(
                f"Epoch {epoch+1} Summary | "
                f"Average Train Loss: {avg_loss:.4f} | "
                f"Eval Loss: {eval_loss:.4f} | "
                f"WER: {wer:.2f} | CER: {cer:.2f} | "
                f"Epoch Time: {timedelta(seconds=int(epoch_time))} | "
                f"Average Batch Time: {avg_batch_time:.2f}s"
            )
            if cfg.wandb.enabled:
                wandb.log({
                    "train/epoch_loss": avg_loss,
                    "eval/epoch_loss": eval_loss,
                    "eval/epoch_wer": wer,
                    "eval/epoch_cer": cer
                })
            
            # Clean up eval model if it was created
            if distributed:
                del eval_model
                torch.cuda.empty_cache()
        
        # Synchronize all processes before starting the next epoch
        if distributed:
            dist.barrier()

@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set up distributed training based on LOCAL_RANK environment variable
    # which is set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    distributed = local_rank != -1
    
    if distributed:
        # Initialize the process group
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        
        # Suppress logging on non-master processes
        if local_rank != 0:
            logger.setLevel(logging.WARNING)
    else:
        # Not in distributed mode, use device 0
        local_rank = 0
        torch.cuda.set_device(local_rank)
    
    # Common performance settings
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.cuda.set_per_process_memory_fraction(0.95, device=torch.cuda.current_device())
    
    # Log the Hydra configuration for reproducibility (only on main process)
    if local_rank == 0:
        logger.info("Hydra Config:\n" + str(cfg))
        logger.info(f"Running on {torch.cuda.device_count()} GPUs with distributed={distributed}")

    # Initialize the model
    model = AudioLLMModel(cfg)
    model.to(f"cuda:{local_rank}")

    # Apply LoRA config
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
    
    # Wrap model with DDP if distributed training is enabled
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    
    # Initialize wandb (only on main process)
    if cfg.wandb.enabled and local_rank == 0:
        setup_wandb(cfg, local_rank)

    # Train the model
    train(model, cfg, local_rank, distributed)
    
    # Clean up distributed process group
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()